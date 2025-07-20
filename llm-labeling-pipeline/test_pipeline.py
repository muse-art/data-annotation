#!/usr/bin/env python3
"""
Test script for the LLM labeling pipeline.
Tests with a few sample songs without requiring API key.
"""

import json
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

def test_data_extraction():
    """Test the data extraction functionality."""
    print("🧪 Testing Data Extraction...")
    
    try:
        from extract_artist_songs import HooktheoryDataExtractor
        
        # Test with sample of 5 songs
        extractor = HooktheoryDataExtractor("../hooktheory_data/song_data")
        extractor.extract_from_directory_structure()
        
        if len(extractor.songs) > 0:
            # Limit to 5 for testing
            extractor.songs = extractor.songs[:5]
            extractor.save_dataset("data/test_dataset.json")
            extractor.print_statistics()
            
            print("✅ Data extraction test passed!")
            return True
        else:
            print("❌ No songs found in dataset")
            return False
            
    except Exception as e:
        print(f"❌ Data extraction test failed: {e}")
        return False

def test_prompt_generation():
    """Test the LLM prompt generation without making API calls."""
    print("\n🧪 Testing Prompt Generation...")
    
    try:
        from llm_labeler import LLMLabeler
        
        # Create a mock labeler (no API key needed for prompt testing)
        class MockLabeler(LLMLabeler):
            def __init__(self):
                # Skip parent init to avoid API key requirement
                pass
        
        labeler = MockLabeler()
        
        # Test prompt generation
        prompt = labeler._create_search_prompt("Porter Robinson", "Look at the Sky")
        
        expected_elements = [
            "Porter Robinson",
            "Look at the Sky", 
            "genres",
            "descriptors",
            "sentence",
            "JSON"
        ]
        
        for element in expected_elements:
            if element not in prompt:
                print(f"❌ Missing expected element: {element}")
                return False
        
        print("✅ Prompt generation test passed!")
        print(f"Sample prompt length: {len(prompt)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Prompt generation test failed: {e}")
        return False

def test_response_parsing():
    """Test the LLM response parsing functionality."""
    print("\n🧪 Testing Response Parsing...")
    
    try:
        from llm_labeler import LLMLabeler, SongLabels
        
        # Create a mock labeler 
        class MockLabeler(LLMLabeler):
            def __init__(self):
                # Skip parent init
                import logging
                self.logger = logging.getLogger(__name__)
        
        labeler = MockLabeler()
        
        # Test valid response
        valid_response = '''```json
{
  "artist": "Porter Robinson",
  "song": "Look at the Sky",
  "genres": [
    "Progressive House",
    "Electropop", 
    "Synthpop"
  ],
  "descriptors": [
    "hopeful",
    "optimistic", 
    "uplifting",
    "emotional"
  ],
  "sentence": "A deeply personal and uplifting electronic anthem about overcoming depression and finding hope for the future."
}
```'''
        
        parsed = labeler._parse_llm_response(valid_response)
        
        if parsed is None:
            print("❌ Failed to parse valid response")
            return False
        
        # Check that we got a SongLabels instance
        if not isinstance(parsed, SongLabels):
            print("❌ Parsed response should be a SongLabels instance")
            return False
        
        # Check required fields using Pydantic model
        if not parsed.artist or not parsed.song:
            print("❌ Missing artist or song")
            return False
        
        # Check types and constraints
        if not parsed.genres or not parsed.descriptors:
            print("❌ Genres and descriptors should not be empty")
            return False
        
        if len(parsed.sentence) < 20:
            print("❌ Sentence should be at least 20 characters")
            return False
        
        print("✅ Response parsing test passed!")
        print(f"Parsed {len(parsed.genres)} genres and {len(parsed.descriptors)} descriptors")
        print(f"Sentence length: {len(parsed.sentence)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Response parsing test failed: {e}")
        return False

def test_pipeline_structure():
    """Test the main pipeline structure."""
    print("\n🧪 Testing Pipeline Structure...")
    
    try:
        from main_pipeline import MusicLabelingPipeline
        
        # Create pipeline without initializing LLM
        pipeline = MusicLabelingPipeline(batch_size=5)
        
        # Test progress tracking
        test_progress = {
            'current_index': 10,
            'total_songs': 100,
            'batch_size': 5
        }
        
        # Test cost estimation (should not crash)
        cost = pipeline._estimate_remaining_cost(10, 100)
        
        if cost <= 0:
            print("❌ Cost estimation should return positive value")
            return False
        
        print("✅ Pipeline structure test passed!")
        print(f"Estimated cost for 90 songs: ${cost:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline structure test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files are present."""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        "scripts/extract_artist_songs.py",
        "scripts/llm_labeler.py", 
        "scripts/main_pipeline.py",
        "requirements.txt",
        "README.md",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ File structure test passed!")
    return True

def main():
    """Run all tests."""
    print("🎵 LLM Music Labeling Pipeline - Test Suite")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_data_extraction,
        test_prompt_generation,
        test_response_parsing,
        test_pipeline_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenRouter API key: export OPENROUTER_API_KEY='your-key-here'")
        print("2. Run a small test: cd llm-labeling-pipeline && python scripts/main_pipeline.py --max-songs 3 --batch-size 3")
        print("3. For full dataset: python scripts/main_pipeline.py")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())