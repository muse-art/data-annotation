# LLM-Based Music Labeling Pipeline

A comprehensive pipeline for automatically generating genre labels, descriptors, and descriptive sentences for MIDI music data using Large Language Models with web search capabilities.

## Overview

This pipeline processes the Hooktheory dataset (28,490+ songs) to generate structured metadata including:
- **Genres**: Musical style classifications (e.g., "Progressive House", "Electropop")
- **Descriptors**: Emotional/tonal qualities (e.g., "hopeful", "nostalgic", "anthemic")
- **Sentences**: Contextual descriptions about the song's meaning and creation

## Directory Structure

```
llm-labeling-pipeline/
├── scripts/           # Core pipeline scripts
├── data/             # Processed datasets and mappings
├── results/          # Generated labels and outputs
├── logs/             # Processing logs and error tracking
└── README.md         # This file
```

## Dataset

**Source**: Hooktheory dataset with 28,490 JSON files containing:
- Artist and song metadata
- MIDI chord progressions and melodies
- Harmonic analysis data
- YouTube video IDs and timing

**Target Output Format**:
```json
{
  "artist": "Porter Robinson",
  "song": "Look at the Sky",
  "genres": ["Progressive House", "Electropop", "Synthpop"],
  "descriptors": ["hopeful", "optimistic", "uplifting", "emotional"],
  "sentence": "A deeply personal and uplifting electronic anthem about overcoming depression and finding hope for the future."
}
```

## Pipeline Components

### 1. Data Extraction (`extract_artist_songs.py`)
- Scans `hooktheory_data/song_data/` folder structure
- Extracts artist/song pairs from file paths and JSON metadata
- Creates master dataset for processing

### 2. LLM Labeling (`llm_labeler.py`)
- Uses OpenRouter API with Perplexity Sonar model with built-in web search capabilities
- Implements structured prompt for consistent output format
- Uses Pydantic models for robust data validation and type safety

### 3. Pipeline Orchestrator (`main_pipeline.py`)
- Batch processing with rate limiting
- Error handling and retry logic
- Progress tracking and resume capability
- Cost monitoring for API usage

### 4. Integration Layer (`pipeline_integration.py`)
- Integrates with existing RYM scraper pipeline
- Provides fallback logic and source prioritization
- Maintains compatibility with current workflow

## Usage

1. **Extract song data**:
```bash
python scripts/extract_artist_songs.py
```

2. **Run LLM labeling**:
```bash
python scripts/main_pipeline.py --batch-size 50 --start-from 0
```

3. **Monitor progress**:
```bash
tail -f logs/pipeline.log
```

## Configuration

- **API Keys**: Set `OPENROUTER_API_KEY` environment variable
- **Model**: Uses Perplexity Sonar (`perplexity/sonar`) by default with built-in web search
- **Rate Limits**: Configurable delays between API calls
- **Batch Size**: Process songs in configurable batches
- **Resume**: Automatic progress saving and resume capability

## Output

- **JSON Results**: Structured labels in `results/` directory
- **CSV Exports**: Human-readable format for analysis
- **Progress Logs**: Detailed processing information
- **Error Tracking**: Failed songs with retry information

## Cost Estimation

- **Reasonable**: Using Perplexity Sonar at $1 per 1M input tokens, $1 per 1M output tokens
- **Built-in Web Search**: No additional fees for web search capabilities
- Estimated ~$157 for full dataset processing (based on actual usage: ~$0.0055 per song)
- Still much cheaper than GPT-4 alternatives with web search fees (~$800+)
- Configurable batch sizes for budget control
- Real-time cost tracking and monitoring