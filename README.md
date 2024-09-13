# Answer_Evaluation

## Overview

This application uses Streamlit to facilitate recording or uploading audio files for technical interview answers. It processes the audio to generate a detailed report, which includes transcription, summary, key points extraction, clarity, relevance, depth evaluation, and sentiment analysis. The report is saved in a `.docx` file that can be downloaded.

## Requirements

- `Python 3.10`
- `streamlit`
- `openai`
- `transformers`
- `torch`
- `docx`
- `audio_recorder_streamlit`
- `python-dotenv`

## Setup

1. **Install Dependencies**:

   ```bash
   pip install streamlit openai transformers torch docx audio_recorder_streamlit python-dotenv
   ```
   or
   ```bash
   pip install -r req_st.txt
   ```

2. **Create a .env File**:  Add your OpenAI API key to a .env file in the root directory:
  
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```
# Code Explanation
## Functions

- **`transcribe_audio()`**: Transcribes audio files using OpenAI's API.
- **`abstract_summary_extraction()`**: Summarizes the transcription.
- **`key_points_extraction()`**: Extracts key points.
- **`evaluate_clarity_of_key_points()`**: Rates clarity of key points.
- **`evaluate_relevance_of_key_points()`**: Rates relevance of key points.
- **`evaluate_depth_of_key_points()`**: Rates depth of key points.
- **`evaluate_sentiment_of_transcription()`**: Performs sentiment analysis.
- **`split_complex_responses()`**: Splits responses into manageable parts.
- **`save_as_docx()`**: Saves the results in a `.docx` file.
- **`process_audio_and_generate_report()`**: Main function to process audio and generate the report.
- **`generate_interview_question()`**: Randomly selects an interview question.

# Running the Application

To run the Streamlit app, use the following command:

```bash
streamlit run streamlit.py
```
