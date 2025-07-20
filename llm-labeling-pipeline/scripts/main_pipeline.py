#!/usr/bin/env python3
"""
Main pipeline orchestrator for LLM-based music labeling.
Handles batch processing, progress tracking, and resume functionality.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import argparse

from llm_labeler import LLMLabeler
from extract_artist_songs import HooktheoryDataExtractor

class MusicLabelingPipeline:
    def __init__(self, 
                 batch_size: int = 10,
                 output_dir: str = "results",
                 model: str = "perplexity/sonar"):
        """Initialize the music labeling pipeline."""
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = model
        self.labeler = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/pipeline.log'),
                logging.StreamHandler()
            ]
        )
        
        # Progress tracking
        self.progress_file = Path("progress.json")
        self.total_processed = 0
        self.start_time = None
        
    def _init_labeler(self):
        """Initialize the LLM labeler."""
        if not self.labeler:
            try:
                self.labeler = LLMLabeler(model=self.model)
                self.logger.info(f"LLM labeler initialized with model: {self.model}")
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM labeler: {e}")
                raise
    
    def _save_progress(self, current_idx: int, total_songs: int, batch_results: List[Dict]):
        """Save current progress to resume later."""
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'current_index': current_idx,
            'total_songs': total_songs,
            'total_processed': self.total_processed,
            'batch_size': self.batch_size,
            'model': self.model,
            'last_batch_size': len(batch_results),
            'success_rate': getattr(self.labeler, 'successful_requests', 0) / max(1, getattr(self.labeler, 'successful_requests', 0) + getattr(self.labeler, 'failed_requests', 0)) * 100 if self.labeler else 0,
            'total_cost': getattr(self.labeler, 'total_cost', 0),
            'estimated_remaining_cost': self._estimate_remaining_cost(current_idx, total_songs),
            'batch_completed': True,
            'last_batch_start_idx': current_idx - self.batch_size,
            'last_batch_end_idx': current_idx - 1
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def _load_progress(self) -> Optional[Dict]:
        """Load previous progress if available."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")
        return None
    
    def _estimate_remaining_cost(self, current_idx: int, total_songs: int) -> float:
        """Estimate cost for remaining songs."""
        if not self.labeler or self.labeler.successful_requests == 0:
            # Default estimate for Perplexity Sonar: ~$0.0055 per song (based on actual usage)
            remaining_songs = total_songs - current_idx
            return remaining_songs * 0.0055
        
        avg_cost_per_song = self.labeler.total_cost / self.labeler.successful_requests
        remaining_songs = total_songs - current_idx
        return remaining_songs * avg_cost_per_song
    
    def _extract_data_if_needed(self, input_path: str, max_songs: Optional[int] = None) -> str:
        """Extract data from Hooktheory if needed."""
        dataset_path = Path("data/artist_song_dataset.json")
        
        if not dataset_path.exists():
            self.logger.info("Dataset not found, extracting from Hooktheory data...")
            
            extractor = HooktheoryDataExtractor(input_path)
            extractor.extract_from_directory_structure()
            
            if max_songs and max_songs < len(extractor.songs):
                extractor.songs = extractor.songs[:max_songs]
                self.logger.info(f"Limited extraction to {max_songs} songs")
            
            extractor.save_dataset(str(dataset_path))
            extractor.print_statistics()
            
        return str(dataset_path)
    
    def run_pipeline(self,
                    input_path: str = "../hooktheory_data/song_data",
                    start_from: int = 0,
                    max_songs: Optional[int] = None,
                    resume: bool = True) -> None:
        """Run the complete labeling pipeline."""
        
        self.start_time = time.time()
        self.logger.info("🎵 Starting Music Labeling Pipeline")
        
        # Check for resume
        if resume:
            progress = self._load_progress()
            if progress:
                self.logger.info(f"Found previous progress: {progress['total_processed']} songs processed")
                start_from = max(start_from, progress['current_index'])
                self.total_processed = progress['total_processed']
        
        # Extract data if needed
        if input_path.endswith('song_data') or Path(input_path).is_dir():
            dataset_path = self._extract_data_if_needed(input_path, max_songs)
        else:
            dataset_path = input_path
        
        # Load dataset
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            songs = dataset.get('songs', [])
            self.logger.info(f"Loaded {len(songs)} songs from dataset")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            return
        
        # Apply limits
        if max_songs and max_songs < len(songs):
            songs = songs[:max_songs]
            self.logger.info(f"Limited to {max_songs} songs")
        
        # Initialize labeler
        self._init_labeler()
        
        # Process in batches
        current_idx = start_from
        batch_num = current_idx // self.batch_size + 1
        
        try:
            while current_idx < len(songs):
                batch_start_time = time.time()
                
                self.logger.info(f"\n📦 Processing Batch {batch_num}")
                self.logger.info(f"Songs {current_idx + 1}-{min(current_idx + self.batch_size, len(songs))} of {len(songs)}")
                
                # Process batch
                batch_results = self.labeler.label_batch(
                    songs=songs,
                    start_idx=current_idx,
                    batch_size=self.batch_size
                )
                
                # Save batch results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_file = self.output_dir / f"batch_{batch_num:03d}_{timestamp}.json"
                
                self.labeler.save_results(batch_results, str(batch_file), batch_num)
                
                # Update progress
                self.total_processed += len(batch_results)
                current_idx += self.batch_size
                
                # Save progress
                self._save_progress(current_idx, len(songs), batch_results)
                
                # Show batch summary
                batch_time = time.time() - batch_start_time
                successful = len([r for r in batch_results if 'labels' in r and 'genres' in r.get('labels', {})])
                failed = len([r for r in batch_results if 'error' in r])
                
                self.logger.info(f"✅ Batch {batch_num} completed in {batch_time:.1f}s")
                self.logger.info(f"   Success: {successful}, Failed: {failed}")
                self.logger.info(f"   Progress: {self.total_processed}/{len(songs)} songs ({self.total_processed/len(songs)*100:.1f}%)")
                
                # Show cost info
                if self.labeler:
                    remaining_cost = self._estimate_remaining_cost(current_idx, len(songs))
                    self.logger.info(f"   Cost so far: ${self.labeler.total_cost:.2f}, Estimated remaining: ${remaining_cost:.2f}")
                
                batch_num += 1
                
                # Break between batches (except for last batch)
                if current_idx < len(songs):
                    self.logger.info("⏸️  Pausing 5 seconds between batches...")
                    time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("\n⚠️ Pipeline interrupted by user")
        
        except Exception as e:
            self.logger.error(f"❌ Pipeline error: {e}")
        
        finally:
            # Final summary
            self._print_final_summary(len(songs))
    
    def _print_final_summary(self, total_songs: int):
        """Print final pipeline summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'='*50}")
        print(f"🎯 PIPELINE SUMMARY")
        print(f"{'='*50}")
        print(f"Total songs in dataset: {total_songs:,}")
        print(f"Songs processed: {self.total_processed:,}")
        print(f"Completion rate: {self.total_processed/total_songs*100:.1f}%")
        print(f"Total time: {total_time/60:.1f} minutes")
        
        if self.labeler:
            print(f"Successful labels: {self.labeler.successful_requests:,}")
            print(f"Failed labels: {self.labeler.failed_requests:,}")
            print(f"Success rate: {self.labeler.successful_requests / max(1, self.labeler.successful_requests + self.labeler.failed_requests) * 100:.1f}%")
            print(f"Total tokens used: {self.labeler.total_tokens:,}")
            print(f"Total cost: ${self.labeler.total_cost:.2f}")
            print(f"Average cost per song: ${self.labeler.total_cost / max(1, self.labeler.successful_requests):.3f}")
        
        print(f"Results saved to: {self.output_dir.absolute()}")
        print(f"{'='*50}")
    
    def _find_processed_song_indices(self) -> set:
        """Find which song indices have already been processed by examining batch files."""
        processed_indices = set()
        batch_files = list(self.output_dir.glob("batch_*.json"))
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    results = batch_data.get('results', [])
                    
                    # Extract indices from original_data file paths or use batch metadata
                    for i, result in enumerate(results):
                        # Try to determine the song index from the batch file and result position
                        # This is a best-effort approach since we don't store indices directly
                        batch_idx = batch_data.get('metadata', {}).get('batch_index', 0)
                        if batch_idx == 1:
                            # First batch started at index 0 but was interrupted
                            processed_indices.add(i)
                        elif batch_idx in [2, 3]:
                            # Batches 2-3 were in the 100s range
                            base_idx = 100 + (batch_idx - 2) * 50
                            processed_indices.add(base_idx + i)
                        elif batch_idx == 4:
                            # Batch 4 was at index 150
                            processed_indices.add(150 + i)
                        elif batch_idx >= 5:
                            # Batches 5+ started at index 200
                            base_idx = 200 + (batch_idx - 5) * 50
                            processed_indices.add(base_idx + i)
                            
            except Exception as e:
                self.logger.warning(f"Could not analyze batch file {batch_file}: {e}")
        
        return processed_indices
    
    def find_missing_song_indices(self, total_songs: int) -> List[int]:
        """Find which song indices have not been processed yet."""
        processed_indices = self._find_processed_song_indices()
        all_indices = set(range(total_songs))
        missing_indices = sorted(all_indices - processed_indices)
        return missing_indices

    def consolidate_results(self, output_file: str = "results/consolidated_labels.json"):
        """Consolidate all batch results into a single file."""
        self.logger.info("Consolidating batch results...")
        
        all_results = []
        batch_files = list(self.output_dir.glob("batch_*.json"))
        batch_files.sort()
        
        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                    results = batch_data.get('results', [])
                    all_results.extend(results)
                    
            except Exception as e:
                self.logger.warning(f"Could not load batch file {batch_file}: {e}")
        
        # Create consolidated output
        consolidated = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_songs': len(all_results),
                'successful_labels': len([r for r in all_results if 'labels' in r and 'genres' in r.get('labels', {})]),
                'failed_labels': len([r for r in all_results if 'error' in r]),
                'batch_files_processed': len(batch_files),
                'model': self.model
            },
            'results': all_results
        }
        
        # Save consolidated file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Consolidated {len(all_results)} results to {output_path}")
        
        # Also create a clean CSV
        csv_path = output_path.with_suffix('.csv')
        self._create_consolidated_csv(all_results, csv_path)
        
        return str(output_path)
    
    def _create_consolidated_csv(self, results: List[Dict], csv_path: Path):
        """Create a clean consolidated CSV file."""
        import csv
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['artist', 'song', 'genres', 'descriptors', 'sentence', 'status', 'file_path']
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
                        'file_path': result.get('original_data', {}).get('file_path', '')
                    })
                elif 'labels' in result:
                    # Handle new Pydantic structure
                    labels = result['labels']
                    writer.writerow({
                        'artist': labels.get('artist', ''),
                        'song': labels.get('song', ''),
                        'genres': '; '.join(labels.get('genres', [])),
                        'descriptors': '; '.join(labels.get('descriptors', [])),
                        'sentence': labels.get('sentence', ''),
                        'status': 'success',
                        'file_path': result.get('original_data', {}).get('file_path', '')
                    })
                else:
                    # Handle legacy structure (fallback)
                    writer.writerow({
                        'artist': result.get('artist', ''),
                        'song': result.get('song', ''),
                        'genres': '; '.join(result.get('genres', [])),
                        'descriptors': '; '.join(result.get('descriptors', [])),
                        'sentence': result.get('sentence', ''),
                        'status': 'success',
                        'file_path': result.get('original_data', {}).get('file_path', '')
                    })
        
        self.logger.info(f"Consolidated CSV saved to {csv_path}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Music Labeling Pipeline with LLM')
    parser.add_argument('--input', default='../hooktheory_data/song_data', 
                       help='Input path (dataset file or song_data directory)')
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='Batch size for processing')
    parser.add_argument('--start-from', type=int, default=0, 
                       help='Index to start from')
    parser.add_argument('--max-songs', type=int, 
                       help='Maximum number of songs to process')
    parser.add_argument('--model', default='perplexity/sonar', 
                       help='Model to use (default: perplexity/sonar)')
    parser.add_argument('--no-resume', action='store_true', 
                       help='Don\'t resume from previous progress')
    parser.add_argument('--consolidate-only', action='store_true',
                       help='Only consolidate existing batch results')
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-api-key-here'")
        return 1
    
    # Create pipeline
    pipeline = MusicLabelingPipeline(
        batch_size=args.batch_size,
        model=args.model
    )
    
    if args.consolidate_only:
        # Just consolidate existing results
        consolidated_file = pipeline.consolidate_results()
        print(f"✅ Results consolidated to {consolidated_file}")
        return 0
    
    # Run the full pipeline
    pipeline.run_pipeline(
        input_path=args.input,
        start_from=args.start_from,
        max_songs=args.max_songs,
        resume=not args.no_resume
    )
    
    # Consolidate results
    if pipeline.total_processed > 0:
        consolidated_file = pipeline.consolidate_results()
        print(f"\n📁 All results consolidated to {consolidated_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())