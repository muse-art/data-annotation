#!/usr/bin/env python3
"""
Extract artist/song pairs from the Hooktheory dataset.
Scans the song_data directory structure and creates a master dataset.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Set
import logging
from urllib.parse import unquote

class HooktheoryDataExtractor:
    def __init__(self, base_path: str = "../hooktheory_data/song_data"):
        """Initialize the data extractor."""
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/extraction.log'),
                logging.StreamHandler()
            ]
        )
        
        self.songs = []
        self.duplicates = set()
        self.errors = []
    
    def _clean_filename(self, filename: str) -> str:
        """Clean and decode filename."""
        # Remove .json extension
        name = filename.replace('.json', '')
        # URL decode if needed
        name = unquote(name)
        # Clean up common issues
        name = name.replace('_', ' ').strip()
        return name
    
    def _clean_artist_name(self, artist_dir: str) -> str:
        """Clean artist directory name."""
        # URL decode and clean
        artist = unquote(artist_dir)
        artist = artist.replace('-', ' ').replace('_', ' ')
        # Handle common patterns
        artist = artist.title()
        return artist.strip()
    
    def _read_song_metadata(self, file_path: Path) -> Dict:
        """Read metadata from a song JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first 10KB to get metadata from the first object
                content = f.read(10000)
                
            # Check if it's a JSON array
            if content.strip().startswith('['):
                # Find the first complete JSON object in the array
                start_idx = content.find('{')
                if start_idx == -1:
                    return {}
                
                # Find the matching closing brace for the first object
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(content[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if brace_count == 0:  # Found complete object
                    json_str = content[start_idx:end_idx]
                    first_object = json.loads(json_str)
                    return first_object
                else:
                    # Object is truncated, try to load the full file if it's not too large
                    file_size = file_path.stat().st_size
                    if file_size < 50000:  # Less than 50KB, safe to load fully
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list) and len(data) > 0:
                                return data[0]
                    return {}
            else:
                # Try to parse as single JSON object
                try:
                    data = json.loads(content)
                    return data if isinstance(data, dict) else {}
                except json.JSONDecodeError:
                    return {}
                
        except Exception as e:
            self.logger.warning(f"Could not read metadata from {file_path}: {e}")
            return {}
    
    def extract_from_directory_structure(self) -> None:
        """Extract artist/song pairs from directory structure."""
        self.logger.info(f"Scanning directory structure at {self.base_path}")
        
        if not self.base_path.exists():
            self.logger.error(f"Base path does not exist: {self.base_path}")
            return
        
        song_count = 0
        
        # Walk through artist directories
        for artist_dir in self.base_path.iterdir():
            if not artist_dir.is_dir():
                continue
                
            artist_name = self._clean_artist_name(artist_dir.name)
            
            # Walk through song files in this artist directory
            for song_file in artist_dir.glob("*.json"):
                song_name = self._clean_filename(song_file.name)
                
                # Create unique identifier
                song_id = f"{artist_name.lower()}|||{song_name.lower()}"
                
                if song_id in self.duplicates:
                    continue
                self.duplicates.add(song_id)
                
                # Try to read additional metadata from the file
                metadata = self._read_song_metadata(song_file)
                
                song_entry = {
                    'artist': artist_name,
                    'song': song_name,
                    'file_path': str(song_file.relative_to(self.base_path.parent)),
                    'artist_dir': artist_dir.name,
                    'song_file': song_file.name,
                    'metadata': {}
                }
                
                # Add metadata if available
                if metadata:
                    song_entry['metadata'] = {
                        'youtube_id': metadata.get('youTubeID'),
                        'song_key': metadata.get('songKey'),
                        'bpm': metadata.get('bpm'),
                        'mode': metadata.get('mode'),
                        'song_url': metadata.get('songURL'),
                        'artist_url': metadata.get('artistURL')
                    }
                
                self.songs.append(song_entry)
                song_count += 1
                
                if song_count % 1000 == 0:
                    self.logger.info(f"Processed {song_count} songs...")
        
        self.logger.info(f"Extracted {len(self.songs)} unique songs from {len(self.duplicates)} total files")
    
    def save_dataset(self, output_path: str = "data/artist_song_dataset.json") -> None:
        """Save the extracted dataset."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = {
            'metadata': {
                'total_songs': len(self.songs),
                'extraction_method': 'directory_structure',
                'source_path': str(self.base_path),
                'duplicate_count': len(self.duplicates) - len(self.songs)
            },
            'songs': self.songs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Dataset saved to {output_file}")
        
        # Also save a simple CSV for quick viewing
        csv_path = output_file.with_suffix('.csv')
        self._save_as_csv(csv_path)
    
    def _save_as_csv(self, csv_path: Path) -> None:
        """Save a simple CSV version for quick viewing."""
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Artist', 'Song', 'YouTube_ID', 'File_Path'])
            
            for song in self.songs:
                writer.writerow([
                    song['artist'],
                    song['song'],
                    song['metadata'].get('youtube_id', ''),
                    song['file_path']
                ])
        
        self.logger.info(f"CSV saved to {csv_path}")
    
    def get_sample_songs(self, count: int = 10) -> List[Dict]:
        """Get a sample of songs for testing."""
        return self.songs[:count] if self.songs else []
    
    def print_statistics(self) -> None:
        """Print dataset statistics."""
        if not self.songs:
            self.logger.warning("No songs extracted")
            return
        
        # Count artists
        artists = set(song['artist'] for song in self.songs)
        
        # Count songs with YouTube IDs
        with_youtube = sum(1 for song in self.songs if song['metadata'].get('youtube_id'))
        
        # Count songs with BPM
        with_bpm = sum(1 for song in self.songs if song['metadata'].get('bpm'))
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Total Songs: {len(self.songs):,}")
        print(f"Unique Artists: {len(artists):,}")
        print(f"Songs with YouTube ID: {with_youtube:,} ({with_youtube/len(self.songs)*100:.1f}%)")
        print(f"Songs with BPM: {with_bpm:,} ({with_bpm/len(self.songs)*100:.1f}%)")
        
        # Show sample artists
        sample_artists = sorted(list(artists))[:10]
        print(f"\nSample Artists: {', '.join(sample_artists)}")
        
        # Show sample songs
        print(f"\nSample Songs:")
        for song in self.songs[:5]:
            print(f"  {song['artist']} - {song['song']}")

def main():
    """Main function to extract the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract artist/song data from Hooktheory dataset')
    parser.add_argument('--base-path', default='../hooktheory_data/song_data',
                       help='Base path to the song_data directory')
    parser.add_argument('--output', default='data/artist_song_dataset.json',
                       help='Output file path')
    parser.add_argument('--sample', type=int, help='Extract only a sample of N songs for testing')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = HooktheoryDataExtractor(args.base_path)
    
    # Extract data
    extractor.extract_from_directory_structure()
    
    # Limit to sample if requested
    if args.sample and args.sample < len(extractor.songs):
        extractor.songs = extractor.songs[:args.sample]
        extractor.logger.info(f"Limited to sample of {args.sample} songs")
    
    # Print statistics
    extractor.print_statistics()
    
    # Save dataset
    extractor.save_dataset(args.output)
    
    print(f"\n✅ Extraction complete! Dataset saved to {args.output}")

if __name__ == "__main__":
    main()