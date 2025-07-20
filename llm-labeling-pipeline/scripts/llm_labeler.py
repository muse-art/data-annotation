#!/usr/bin/env python3
"""
LLM-based music labeling using web search capabilities.
Generates structured genre, descriptor, and sentence data for songs.
"""

import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
import re
import openai
from openai import OpenAI
from datetime import datetime
import requests
from pydantic import BaseModel, Field, validator

class SongLabels(BaseModel):
    """Pydantic model for song label validation."""
    artist: str = Field(..., description="Artist name")
    song: str = Field(..., description="Song title")
    genres: List[str] = Field(..., min_items=1, max_items=8, description="List of music genres")
    descriptors: List[str] = Field(..., min_items=1, max_items=40, description="List of emotion, mood, style, instrumentation, and other descriptors")
    sentence: str = Field(..., min_length=20, max_length=500, description="Descriptive sentence about the compositional characteristics of the song. Avoid explicity mention of the artist or song name. ")
    
    @validator('genres')
    def validate_genres(cls, v):
        """Validate genres are non-empty strings."""
        if not all(isinstance(genre, str) and genre.strip() for genre in v):
            raise ValueError("All genres must be non-empty strings")
        return [genre.strip() for genre in v]
    
    @validator('descriptors')
    def validate_descriptors(cls, v):
        """Validate descriptors are non-empty strings."""
        if not all(isinstance(desc, str) and desc.strip() for desc in v):
            raise ValueError("All descriptors must be non-empty strings")
        return [desc.strip() for desc in v]
    
    @validator('sentence')
    def validate_sentence(cls, v):
        """Validate sentence is meaningful."""
        if not v.strip():
            raise ValueError("Sentence cannot be empty")
        return v.strip()

class LabelingMetadata(BaseModel):
    """Metadata for labeling results."""
    timestamp: str
    model: str
    tokens_used: int
    estimated_cost: float
    prompt_tokens: int
    completion_tokens: int

class LabelingResult(BaseModel):
    """Complete labeling result with metadata."""
    labels: SongLabels
    metadata: LabelingMetadata
    original_data: Dict

class LLMLabeler:
    def __init__(self, api_key: Optional[str] = None, model: str = "perplexity/sonar"):
        """Initialize the LLM labeler."""
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        self.model = model
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/llm_labeling.log'),
                logging.StreamHandler()
            ]
        )
        
        # Tracking for costs and performance
        self.total_tokens = 0
        self.total_cost = 0.0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds between requests
    
    def _rate_limit(self):
        """Implement rate limiting to respect API limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate API cost based on token usage."""
        # Perplexity Sonar actual costs based on real usage data: ~$0.0055 per request
        # This accounts for the model's pricing plus any additional fees
        # Using a fixed per-request cost since actual costs don't correlate directly with token count
        return 0.0055
    
    def _create_search_prompt(self, artist: str, song: str) -> str:
        """Create the search prompt for the LLM."""
        return f"""Please search the web for information about the song "{song}" by {artist} and provide a detailed analysis in the following JSON format:

{{
  "artist": "{artist}",
  "song": "{song}",
  "genres": [
    "genre1",
    "genre2",
    "genre3"
  ],
  "descriptors": [
    "descriptor1",
    "descriptor2", 
    "descriptor3",
    "descriptor4"
  ],
  "sentence": "A descriptive sentence about the song's compositional characteristics and musical elements."
}}

Search Guidelines:
- Search for the song on music databases, streaming platforms, and music review sites
- Look for information about the song's genre classification, musical style, and production details
- Find reviews or analyses that describe the song's mood, instrumentation, and compositional elements
- Research the artist's typical style if the specific song information is limited

Output Requirements:
- Include 4-8 relevant genres (e.g., "Progressive House", "Electropop", "Synthpop", "Alternative Rock")
- Include 5-40 descriptors focusing on mood, style, and musical characteristics (e.g., "hopeful", "nostalgic", "anthemic", "melancholic", "driving", "atmospheric", "synth-heavy")
- Write a 1-2 sentence description capturing the song's compositional aspects like rhythmic patterns, tonal qualities, instrumentation style, and overall musical character
- Avoid explicitly mentioning the artist or song name in the sentence
- Avoid using technical music theory terms like "chords", "melody", "harmony" directly in the sentence
- Base your analysis on web search results for accuracy

Song to analyze: "{artist} - {song}"

Only respond with the JSON object, no additional text."""

    def _parse_llm_response(self, response_text: str) -> Optional[SongLabels]:
        """Parse and validate the LLM response using Pydantic."""
        try:
            # Clean the response - remove any markdown formatting
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # Parse JSON
            data = json.loads(cleaned_text)
            
            # Validate using Pydantic model
            song_labels = SongLabels(**data)
            
            self.logger.debug(f"Successfully parsed and validated response for {song_labels.artist} - {song_labels.song}")
            return song_labels
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {e}")
            self.logger.error(f"Response text: {response_text}")
            return None
        except ValueError as e:
            self.logger.error(f"Pydantic validation error: {e}")
            self.logger.error(f"Response text: {response_text}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {e}")
            return None
    
    def label_song(self, artist: str, song: str, original_data: Dict = None) -> Optional[LabelingResult]:
        """Label a single song using LLM with web search."""
        self.logger.info(f"Labeling: {artist} - {song}")
        
        # Rate limiting
        self._rate_limit()
        
        prompt = self._create_search_prompt(artist, song)
        
        try:
            # Make API request with OpenRouter headers
            # Perplexity Sonar has built-in web search capabilities
            response = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/your-repo/llm-labeling-pipeline",
                    "X-Title": "Music Labeling Pipeline",
                },
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a music expert with access to web search. Search the web for accurate information about songs and provide well-researched details in the exact JSON format requested. ALWAYS respond with valid JSON only, no additional text."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent formatting
                max_tokens=800
            )
            
            # Track usage
            usage = response.usage
            self.total_tokens += usage.total_tokens
            cost = self._estimate_cost(usage.prompt_tokens, usage.completion_tokens)
            self.total_cost += cost
            
            # Parse response
            response_text = response.choices[0].message.content
            song_labels = self._parse_llm_response(response_text)
            
            if song_labels:
                self.successful_requests += 1
                self.logger.info(f"Success: {len(song_labels.genres)} genres, {len(song_labels.descriptors)} descriptors")
                
                # Create metadata
                metadata = LabelingMetadata(
                    timestamp=datetime.now().isoformat(),
                    model=self.model,
                    tokens_used=usage.total_tokens,
                    estimated_cost=cost,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
                
                # Create complete result
                result = LabelingResult(
                    labels=song_labels,
                    metadata=metadata,
                    original_data=original_data or {}
                )
                
                return result
            else:
                self.failed_requests += 1
                self.logger.error(f"Failed to parse response for {artist} - {song}")
                return None
                
        except Exception as e:
            self.failed_requests += 1
            self.logger.error(f"API request failed for {artist} - {song}: {e}")
            return None
    
    def label_batch(self, songs: List[Dict], start_idx: int = 0, batch_size: int = 10) -> List[Dict]:
        """Label a batch of songs."""
        results = []
        
        end_idx = min(start_idx + batch_size, len(songs))
        batch_songs = songs[start_idx:end_idx]
        
        self.logger.info(f"Labeling batch: {start_idx} to {end_idx-1} ({len(batch_songs)} songs)")
        
        for i, song_data in enumerate(batch_songs):
            artist = song_data['artist']
            song = song_data['song']
            
            try:
                result = self.label_song(artist, song, song_data)
                
                if result:
                    # Convert Pydantic model to dict for JSON serialization
                    result_dict = result.dict()
                    results.append(result_dict)
                else:
                    # Add failed entry
                    failed_entry = {
                        'artist': artist,
                        'song': song,
                        'error': 'Failed to generate labels',
                        'original_data': song_data,
                        'metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'status': 'failed'
                        }
                    }
                    results.append(failed_entry)
                
                # Progress update
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Batch progress: {i + 1}/{len(batch_songs)} songs completed")
                    self.print_usage_stats()
                
            except KeyboardInterrupt:
                self.logger.info("Batch interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error processing {artist} - {song}: {e}")
                continue
        
        self.logger.info(f"Batch completed: {len(results)} results")
        return results
    
    def print_usage_stats(self):
        """Print current usage statistics."""
        success_rate = (self.successful_requests / max(1, self.successful_requests + self.failed_requests)) * 100
        
        print(f"\n=== LLM Usage Statistics ===")
        print(f"Successful requests: {self.successful_requests}")
        print(f"Failed requests: {self.failed_requests}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Estimated total cost: ${self.total_cost:.2f}")
        print(f"Average cost per request: ${self.total_cost / max(1, self.successful_requests):.3f}")
    
    def save_results(self, results: List[Dict], output_path: str, batch_idx: int = 0):
        """Save results to JSON file."""
        from pathlib import Path
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive output structure
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'batch_index': batch_idx,
                'total_songs': len(results),
                'successful_labels': len([r for r in results if 'labels' in r and 'genres' in r.get('labels', {})]),
                'failed_labels': len([r for r in results if 'error' in r]),
                'total_tokens': self.total_tokens,
                'estimated_cost': self.total_cost,
                'success_rate': (self.successful_requests / max(1, self.successful_requests + self.failed_requests)) * 100
            },
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {output_file}")
        
        # Also save CSV for easy viewing
        csv_path = output_file.with_suffix('.csv')
        self._save_as_csv(results, csv_path)
    
    def _save_as_csv(self, results: List[Dict], csv_path):
        """Save results as CSV for easy viewing."""
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['artist', 'song', 'genres', 'descriptors', 'sentence', 'status', 'cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                if 'error' in result:
                    writer.writerow({
                        'artist': result['artist'],
                        'song': result['song'],
                        'genres': '',
                        'descriptors': '',
                        'sentence': '',
                        'status': 'failed',
                        'cost': ''
                    })
                elif 'labels' in result:
                    # Handle new Pydantic structure
                    labels = result['labels']
                    metadata = result.get('metadata', {})
                    
                    writer.writerow({
                        'artist': labels.get('artist', ''),
                        'song': labels.get('song', ''),
                        'genres': '; '.join(labels.get('genres', [])),
                        'descriptors': '; '.join(labels.get('descriptors', [])),
                        'sentence': labels.get('sentence', ''),
                        'status': 'success',
                        'cost': f"${metadata.get('estimated_cost', 0):.3f}"
                    })
                else:
                    # Handle legacy structure (fallback)
                    metadata = result.get('metadata', {})
                    writer.writerow({
                        'artist': result.get('artist', ''),
                        'song': result.get('song', ''),
                        'genres': '; '.join(result.get('genres', [])),
                        'descriptors': '; '.join(result.get('descriptors', [])),
                        'sentence': result.get('sentence', ''),
                        'status': 'success',
                        'cost': f"${metadata.get('estimated_cost', 0):.3f}"
                    })
        
        self.logger.info(f"CSV saved to {csv_path}")

def main():
    """Main function for testing the LLM labeler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-based music labeling')
    parser.add_argument('--input', default='data/artist_song_dataset.json', help='Input dataset file')
    parser.add_argument('--output', default='results/llm_labels.json', help='Output file')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--start-from', type=int, default=0, help='Index to start from')
    parser.add_argument('--max-songs', type=int, help='Maximum number of songs to process')
    parser.add_argument('--model', default='perplexity/sonar', help='Model to use (default: perplexity/sonar)')
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Load input data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        songs = dataset.get('songs', [])
        print(f"Loaded {len(songs)} songs from {args.input}")
        
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found")
        return
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Limit songs if specified
    if args.max_songs:
        songs = songs[:args.max_songs]
        print(f"Limited to {len(songs)} songs")
    
    # Create labeler
    labeler = LLMLabeler(model=args.model)
    
    # Process batch
    try:
        results = labeler.label_batch(
            songs=songs,
            start_idx=args.start_from,
            batch_size=args.batch_size
        )
        
        # Save results
        labeler.save_results(results, args.output)
        
        # Print final stats
        labeler.print_usage_stats()
        
        print(f"\n✅ Labeling complete! Results saved to {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    main()