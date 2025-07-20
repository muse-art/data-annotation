#!/usr/bin/env python3
"""
Script to identify and process missing songs from gaps in pipeline processing.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Set
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from main_pipeline import MusicLabelingPipeline

def find_processed_songs_from_results() -> Set[str]:
    """Find which songs have been processed by examining the consolidated results."""
    processed_songs = set()
    results_dir = Path("results")
    
    # Check consolidated results first
    consolidated_file = results_dir / "consolidated_labels.json"
    if consolidated_file.exists():
        with open(consolidated_file, 'r') as f:
            data = json.load(f)
            for result in data.get('results', []):
                if 'labels' in result:
                    artist = result['labels'].get('artist', '')
                    song = result['labels'].get('song', '')
                elif 'artist' in result and 'song' in result:
                    artist = result['artist']
                    song = result['song']
                else:
                    continue
                processed_songs.add(f"{artist}|||{song}")
    
    return processed_songs

def find_missing_songs(dataset_path: str, max_songs: int = 1000) -> List[dict]:
    """Find songs that haven't been processed yet."""
    # Load the full dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    songs = dataset.get('songs', [])[:max_songs]
    processed_songs = find_processed_songs_from_results()
    
    missing_songs = []
    for i, song in enumerate(songs):
        song_key = f"{song['artist']}|||{song['song']}"
        if song_key not in processed_songs:
            song_with_index = song.copy()
            song_with_index['original_index'] = i
            missing_songs.append(song_with_index)
    
    return missing_songs

def main():
    parser = argparse.ArgumentParser(description='Process missing songs from pipeline gaps')
    parser.add_argument('--dataset', default='data/artist_song_dataset.json',
                       help='Path to the dataset file')
    parser.add_argument('--max-songs', type=int, default=1000,
                       help='Maximum songs that should have been processed')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for processing missing songs')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only identify missing songs, don\'t process them')
    parser.add_argument('--model', default='perplexity/sonar',
                       help='Model to use for processing')
    
    args = parser.parse_args()
    
    # Check API key
    if not args.dry_run and not os.getenv('OPENROUTER_API_KEY'):
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        return 1
    
    print("🔍 Analyzing processed songs...")
    missing_songs = find_missing_songs(args.dataset, args.max_songs)
    
    print(f"\n📊 Analysis Results:")
    print(f"Total songs that should have been processed: {args.max_songs}")
    print(f"Missing songs found: {len(missing_songs)}")
    
    if missing_songs:
        print(f"\n📋 Missing song ranges:")
        # Group consecutive indices for better display
        indices = [song['original_index'] for song in missing_songs]
        ranges = []
        start = indices[0]
        end = start
        
        for i in range(1, len(indices)):
            if indices[i] == end + 1:
                end = indices[i]
            else:
                ranges.append((start, end))
                start = indices[i]
                end = start
        ranges.append((start, end))
        
        for start, end in ranges:
            if start == end:
                print(f"  - Song {start + 1} (index {start})")
            else:
                print(f"  - Songs {start + 1}-{end + 1} (indices {start}-{end})")
    
    if args.dry_run:
        print(f"\n✅ Dry run complete. Found {len(missing_songs)} missing songs.")
        return 0
    
    if not missing_songs:
        print("\n✅ No missing songs found!")
        return 0
    
    # Process missing songs
    print(f"\n🚀 Processing {len(missing_songs)} missing songs...")
    
    # Create a temporary dataset with just the missing songs
    temp_dataset = {
        'metadata': {
            'total_songs': len(missing_songs),
            'source': 'missing_songs_recovery',
            'original_indices': [song['original_index'] for song in missing_songs]
        },
        'songs': missing_songs
    }
    
    temp_dataset_path = 'data/missing_songs_dataset.json'
    with open(temp_dataset_path, 'w') as f:
        json.dump(temp_dataset, f, indent=2)
    
    # Create pipeline and process
    pipeline = MusicLabelingPipeline(
        batch_size=args.batch_size,
        model=args.model
    )
    
    # Process the missing songs
    pipeline.run_pipeline(
        input_path=temp_dataset_path,
        start_from=0,
        max_songs=len(missing_songs),
        resume=False  # Don't resume since this is a clean run
    )
    
    # Clean up temp file
    os.remove(temp_dataset_path)
    
    print(f"\n✅ Finished processing missing songs!")
    print("💡 Run consolidation to update the consolidated results.")
    
    return 0

if __name__ == "__main__":
    exit(main()) 