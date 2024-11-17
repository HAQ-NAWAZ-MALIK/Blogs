# Dataset Processing Script

This script processes a dataset of paired text and audio files, creating a clean dataset and preparing it for use with the Hugging Face `datasets` library. It's particularly useful for text-to-speech (TTS) datasets but can be adapted for other audio-text paired datasets.

## Features

- Extracts tar.gz files
- Cleans and organizes dataset files
- Splits dataset into train and test sets
- Creates metadata for the dataset
- Prepares the dataset for use with Hugging Face's `datasets` library

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- datasets
- soundfile

Install the required packages using:

```
pip install pandas scikit-learn datasets soundfile
```

## Usage

1. Clone this repository or download the script.

2. Modify the `main()` function call at the bottom of the script with your specific paths and dataset details:

```python
if __name__ == "__main__":
    main(
        source_tar='path/to/your/dataset.tar.gz',
        extract_dir='path/to/extracted_dataset',
        clean_dir='path/to/clean_dataset',
        output_dir='path/to/hf_dataset',
        dataset_name='Your Dataset Name',
        language='Your Language'
    )
```

3. Run the script:

```
python dataset_processing_script.py
```

## Script Workflow

1. Extracts the tar.gz file containing the dataset
2. Creates a clean dataset by copying and verifying audio files
3. Splits the dataset into train and test sets
4. Creates metadata for the dataset
5. Prepares and saves the dataset in Hugging Face's dataset format

## Applications

This script is useful for:

1. **TTS Dataset Preparation**: Ideal for preparing text-to-speech datasets with paired audio and text files.

2. **Audio Classification Datasets**: Can be adapted for datasets used in audio classification tasks.

3. **Speech Recognition**: Useful for preparing datasets for automatic speech recognition (ASR) tasks.

4. **Multilingual Dataset Processing**: Can handle datasets in various languages by specifying the language in the metadata.

5. **Dataset Standardization**: Helps in standardizing datasets from different sources into a common format.

6. **Hugging Face Integration**: Prepares datasets for easy use with Hugging Face's `datasets` library and related tools.

## Customization

- Modify the `create_clean_dataset()` function to handle different file formats or naming conventions.
- Adjust the train-test split ratio in the `split_dataset()` function.
- Customize the metadata fields in the `create_metadata()` function to suit your dataset's specific needs.

## License

This script is provided under the MIT License. Feel free to modify and use it for your projects.









## CODE FOR USE

```

import os
import tarfile
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Audio
import json
import soundfile as sf

def extract_tarfile(tar_path, extract_path):
    # Extract the contents of a tar.gz file
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted {tar_path} to {extract_path}")

def create_clean_dataset(source_dir, dest_dir):
    # Create a directory for the clean dataset
    os.makedirs(dest_dir, exist_ok=True)
    
    data = []
    # Iterate through files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith('.txt'):
            base_name = os.path.splitext(filename)[0]
            txt_path = os.path.join(source_dir, filename)
            wav_path = os.path.join(source_dir, f"{base_name}.wav")
            
            # Check if corresponding wav file exists
            if os.path.exists(wav_path):
                # Read the text content
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                # Define new paths for cleaned files
                new_txt_path = os.path.join(dest_dir, filename)
                new_wav_path = os.path.join(dest_dir, f"{base_name}.wav")
                
                # Copy files to the clean directory
                shutil.copy2(txt_path, new_txt_path)
                shutil.copy2(wav_path, new_wav_path)
                
                # Verify audio file and get its properties
                try:
                    audio_info = sf.info(new_wav_path)
                    duration = audio_info.duration
                    sample_rate = audio_info.samplerate
                except Exception as e:
                    print(f"Error processing {new_wav_path}: {str(e)}")
                    continue
                
                # Append file information to the data list
                data.append({
                    'file_name': base_name,
                    'text': text,
                    'audio': new_wav_path,
                    'duration': duration,
                    'sample_rate': sample_rate
                })
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    print(f"Total files processed: {len(df)}")
    print(f"Total audio duration: {df['duration'].sum():.2f} seconds")
    return df

def split_dataset(df, test_size=0.2, random_state=42):
    # Split the dataset into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def create_metadata(df, output_file, dataset_name, language):
    # Create metadata for the dataset
    metadata = {
        'dataset_name': dataset_name,
        'language': language,
        'num_samples': len(df),
        'total_audio_duration': f"{df['duration'].sum():.2f} seconds",
        'file_format': 'WAV',
        'text_format': 'TXT',
        'license': 'CC-BY-4.0',  # Adjust as needed
        'citations': [],  # Add any relevant citations
        'description': f'A {language} text-to-speech dataset containing paired text and audio files.'
    }
    
    # Write metadata to a JSON file
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def create_hf_dataset(train_df, test_df, output_dir):
    def process_example(example):
        # Process each example in the dataset
        return {
            'audio': example['audio'],
            'text': example['text'],
            'duration': example['duration'],
            'sample_rate': example['sample_rate']
        }

    # Create datasets from DataFrames
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Apply processing to each example
    train_dataset = train_dataset.map(process_example)
    test_dataset = test_dataset.map(process_example)
    
    # Set the audio column format
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Create a DatasetDict with train and test splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    # Save the dataset to disk
    dataset_dict.save_to_disk(output_dir)
    print(f"Saved dataset to {output_dir}")

def main(source_tar, extract_dir, clean_dir, output_dir, dataset_name, language):
    # Extract the tar file
    extract_tarfile(source_tar, extract_dir)
    
    # Create a clean dataset
    df = create_clean_dataset(extract_dir, clean_dir)
    
    # Split the dataset into train and test sets
    train_df, test_df = split_dataset(df)
    
    # Create metadata for the dataset
    create_metadata(df, os.path.join(clean_dir, 'metadata.json'), dataset_name, language)
    
    # Create and save the Hugging Face dataset
    create_hf_dataset(train_df, test_df, output_dir)
    
    print("Processing complete!")

if __name__ == "__main__":
    # Example usage (replace these with your actual paths and details)
    main(
        source_tar='path/to/your/dataset.tar.gz',
        extract_dir='path/to/extracted_dataset',
        clean_dir='path/to/clean_dataset',
        output_dir='path/to/hf_dataset',
        dataset_name='Your Dataset Name',
        language='Your Language'
    ) 
```


## If you have .wav files and .txt files in 2 seprated folders you can use it as 


Below is the **complete code** that processes `.wav` files and their corresponding transcriptions into a Hugging Face-compatible dataset, including all features (`duration`, `sample_rate`, metadata, train/test splits) and fixes the issue with non-numeric filenames. 

---

### Complete Code:

```python
import os
import json
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Audio

# Paths
wav_folder = '/content/wavs'
text_folder = '/content/metadata_text_files'
output_dir = '/content/hf_dataset'
metadata_file = '/content/metadata.json'

# Dataset Information
dataset_name = "Persian_TTS"
language = "Persian"

# Step 1: Create a Clean Dataset
def create_clean_dataset(wav_folder, text_folder):
    data = []
    total_duration = 0
    
    # Filter and sort valid .wav files
    wav_files = [
        wav_file for wav_file in os.listdir(wav_folder)
        if wav_file.endswith('.wav') and wav_file.split('.')[0].isdigit()
    ]
    wav_files = sorted(wav_files, key=lambda x: int(x.split('.')[0]))  # Sort by numeric value

    for wav_file in wav_files:
        # Extract WAV file path
        wav_path = os.path.join(wav_folder, wav_file)
        base_name = wav_file.split('.')[0]
        
        # Extract corresponding text file path
        text_file = os.path.join(text_folder, f"wav_{base_name}.txt")
        
        if not os.path.exists(text_file):
            print(f"Skipping {wav_path}: No transcription found.")
            continue
        
        # Read transcription
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Verify audio file and get its properties
        try:
            audio_info = sf.info(wav_path)
            duration = audio_info.duration
            sample_rate = audio_info.samplerate
            total_duration += duration
        except Exception as e:
            print(f"Error processing {wav_path}: {str(e)}")
            continue
        
        # Append file information to the data list
        data.append({
            'file_name': wav_file,
            'text': text,
            'audio': wav_path,
            'duration': duration,
            'sample_rate': sample_rate
        })
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    print(f"Total files processed: {len(df)}")
    print(f"Total audio duration: {total_duration:.2f} seconds")
    return df

# Step 2: Split Dataset
def split_dataset(df, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

# Step 3: Create Metadata
def create_metadata(df, output_file, dataset_name, language):
    metadata = {
        'dataset_name': dataset_name,
        'language': language,
        'num_samples': len(df),
        'total_audio_duration': f"{df['duration'].sum():.2f} seconds",
        'file_format': 'WAV',
        'text_format': 'TXT',
        'license': 'CC-BY-4.0',  # Adjust as needed
        'citations': [],  # Add any relevant citations
        'description': f'A {language} text-to-speech dataset containing paired text and audio files.'
    }
    
    # Write metadata to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {output_file}")

# Step 4: Create Hugging Face Dataset
def create_hf_dataset(train_df, test_df, output_dir):
    def process_example(example):
        # Process each example in the dataset
        return {
            'audio': example['audio'],
            'text': example['text'],
            'duration': example['duration'],
            'sample_rate': example['sample_rate']
        }

    # Create datasets from DataFrames
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Apply processing to each example
    train_dataset = train_dataset.map(process_example)
    test_dataset = test_dataset.map(process_example)
    
    # Set the audio column format
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Create a DatasetDict with train and test splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    # Save the dataset to disk
    dataset_dict.save_to_disk(output_dir)
    print(f"Saved dataset to {output_dir}")

# Main Function
def main():
    # Create a clean dataset
    df = create_clean_dataset(wav_folder, text_folder)
    
    # Split the dataset into train and test sets
    train_df, test_df = split_dataset(df)
    
    # Create metadata for the dataset
    create_metadata(df, metadata_file, dataset_name, language)
    
    # Create and save the Hugging Face dataset
    create_hf_dataset(train_df, test_df, output_dir)
    print("Processing complete!")

# Run the script
main()
```

---

### Key Features:
1. **Handles Non-Numeric Filenames:**
   - Filters files to process only `.wav` files with numeric names.

2. **Includes Audio Features:**
   - Extracts `duration` and `sample_rate` for each `.wav` file.

3. **Creates Metadata:**
   - Saves metadata like the total number of samples, total audio duration, dataset name, and language in a JSON file.

4. **Train/Test Splits:**
   - Splits data into `train` and `test` sets with an 80/20 ratio.

5. **Hugging Face Dataset:**
   - Creates a Hugging Face `DatasetDict` and saves it to disk for reuse.

---

### Output:
1. **Hugging Face Dataset:**
   - Saved to `/content/hf_dataset`.

2. **Metadata JSON:**
   - Saved to `/content/metadata.json`.

3. **Console Logs:**
   - Number of files processed.
   - Total audio duration.

------




























Your clean dataset is being created now push to hub 


```

from datasets import load_from_disk
from huggingface_hub import HfApi, login
import os

def upload_to_hub(dataset_path, repo_name, token):
    # Login to Hugging Face
    login(token)

    # Load the dataset
    dataset = load_from_disk(dataset_path)

    # Push to hub
    dataset.push_to_hub(repo_name, token=token)

    print(f"Dataset uploaded successfully to {repo_name}")

def main():
    # Set your Hugging Face token
    hf_token = "your_hugging_face_token_here"  # Replace with your actual token

    # Set the path to your processed dataset
    dataset_path = "/content/kashmiri_tts_hf_dataset"  Replace with your actual path where your saved the cleaned  Hf data

    # Set the name for your dataset repository on Hugging Face
    repo_name = "Omarrran/kashmiri_tts_hf_dataset"  # Replace with your actual rep

    # Upload the dataset
    upload_to_hub(dataset_path, repo_name, hf_token)

if __name__ == "__main__":
    main()

```


Thank you
