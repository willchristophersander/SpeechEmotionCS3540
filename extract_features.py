#!/usr/bin/env python3
"""
Vocal Feature Extraction Pipeline for CREMA-D Dataset
Extracts comprehensive audio features for emotion classification
"""

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class VocalFeatureExtractor:
    """Extract comprehensive vocal features from audio files"""
    
    def __init__(self, sample_rate=22050, hop_length=512, n_mfcc=13):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        
    def extract_mfcc_features(self, y, sr):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        
        features = {}
        for i in range(self.n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i+1}_min'] = np.min(mfccs[i])
            features[f'mfcc_{i+1}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i+1}_skew'] = skew(mfccs[i])
            features[f'mfcc_{i+1}_kurtosis'] = kurtosis(mfccs[i])
        
        return features
    
    def extract_spectral_features(self, y, sr):
        """Extract spectral features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_skew'] = skew(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_skew'] = skew(zcr)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        for i in range(12):
            features[f'chroma_{i+1}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i+1}_std'] = np.std(chroma[i])
        
        # Tonnetz features
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for i in range(6):
            features[f'tonnetz_{i+1}_mean'] = np.mean(tonnetz[i])
            features[f'tonnetz_{i+1}_std'] = np.std(tonnetz[i])
        
        return features
    
    def extract_rhythm_features(self, y, sr):
        """Extract rhythm and tempo features"""
        features = {}
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        features['tempo'] = tempo
        
        # Rhythm patterns
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
        features['onset_count'] = len(onset_frames)
        features['onset_rate'] = len(onset_frames) / (len(y) / sr)
        
        return features
    
    def extract_energy_features(self, y, sr):
        """Extract energy-related features"""
        features = {}
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_skew'] = skew(rms)
        
        # Energy entropy
        energy_entropy = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        features['energy_entropy_mean'] = np.mean(energy_entropy)
        features['energy_entropy_std'] = np.std(energy_entropy)
        
        return features
    
    def extract_harmonic_features(self, y, sr):
        """Extract harmonic and percussive features"""
        features = {}
        
        # Harmonic and percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Harmonic features
        harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr, hop_length=self.hop_length)[0]
        features['harmonic_centroid_mean'] = np.mean(harmonic_centroid)
        features['harmonic_centroid_std'] = np.std(harmonic_centroid)
        
        # Percussive features
        percussive_centroid = librosa.feature.spectral_centroid(y=y_percussive, sr=sr, hop_length=self.hop_length)[0]
        features['percussive_centroid_mean'] = np.mean(percussive_centroid)
        features['percussive_centroid_std'] = np.std(percussive_centroid)
        
        # Harmonic to noise ratio
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        features['harmonic_to_noise_ratio'] = harmonic_energy / (percussive_energy + 1e-8)
        
        return features
    
    def extract_prosodic_features(self, y, sr):
        """Extract prosodic features (pitch, formants)"""
        features = {}
        
        # Pitch (F0) using YIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), 
                                                   fmax=librosa.note_to_hz('C7'), 
                                                   sr=sr, hop_length=self.hop_length)
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features['f0_mean'] = np.mean(f0_clean)
            features['f0_std'] = np.std(f0_clean)
            features['f0_min'] = np.min(f0_clean)
            features['f0_max'] = np.max(f0_clean)
            features['f0_skew'] = skew(f0_clean)
            features['f0_kurtosis'] = kurtosis(f0_clean)
            features['f0_range'] = np.max(f0_clean) - np.min(f0_clean)
        else:
            # Default values if no pitch detected
            features.update({
                'f0_mean': 0, 'f0_std': 0, 'f0_min': 0, 'f0_max': 0,
                'f0_skew': 0, 'f0_kurtosis': 0, 'f0_range': 0
            })
        
        # Voicing ratio
        features['voicing_ratio'] = np.mean(voiced_flag)
        
        return features
    
    def extract_all_features(self, audio_path):
        """Extract all features from an audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Initialize features dictionary
            features = {}
            
            # Extract different feature groups
            features.update(self.extract_mfcc_features(y, sr))
            features.update(self.extract_spectral_features(y, sr))
            features.update(self.extract_rhythm_features(y, sr))
            features.update(self.extract_energy_features(y, sr))
            features.update(self.extract_harmonic_features(y, sr))
            features.update(self.extract_prosodic_features(y, sr))
            
            # Add basic audio properties
            features['duration'] = len(y) / sr
            features['sample_rate'] = sr
            features['file_path'] = audio_path
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

def parse_filename(filename):
    """Parse CREMA-D filename to extract metadata"""
    # Remove extension
    name = filename.split('.')[0]
    parts = name.split('_')
    
    if len(parts) >= 4:
        actor_id = parts[0]
        sentence = parts[1]
        emotion = parts[2]
        intensity = parts[3]
        
        # Map emotion codes to full names
        emotion_map = {
            'ANG': 'Anger',
            'DIS': 'Disgust', 
            'FEA': 'Fear',
            'HAP': 'Happy',
            'NEU': 'Neutral',
            'SAD': 'Sad'
        }
        
        # Map intensity codes
        intensity_map = {
            'LO': 'Low',
            'MD': 'Medium', 
            'HI': 'High',
            'XX': 'Unspecified'
        }
        
        return {
            'actor_id': actor_id,
            'sentence': sentence,
            'emotion': emotion_map.get(emotion, emotion),
            'intensity': intensity_map.get(intensity, intensity),
            'filename': filename
        }
    
    return None

def main():
    """Main function to extract features from CREMA-D dataset"""
    
    # Paths
    audio_dir = "/Users/will/Projects/SpeechEmotionCS3540/CREMA-D/AudioWAV"
    output_dir = "/Users/will/Projects/SpeechEmotionCS3540/features"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature extractor
    extractor = VocalFeatureExtractor()
    
    # Get all audio files
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    print(f"Found {len(audio_files)} audio files")
    
    # Extract features
    all_features = []
    metadata = []
    
    print("Extracting features...")
    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(audio_dir, audio_file)
        
        # Extract features
        features = extractor.extract_all_features(audio_path)
        
        if features is not None:
            all_features.append(features)
            
            # Parse filename for metadata
            meta = parse_filename(audio_file)
            if meta:
                metadata.append(meta)
    
    # Create DataFrames
    features_df = pd.DataFrame(all_features)
    metadata_df = pd.DataFrame(metadata)
    
    # Merge features with metadata
    if not features_df.empty and not metadata_df.empty:
        # Use filename as key to merge
        features_df['filename'] = features_df['file_path'].apply(lambda x: os.path.basename(x))
        metadata_df['filename'] = metadata_df['filename']
        
        # Merge on filename
        combined_df = features_df.merge(metadata_df, on='filename', how='inner')
        
        # Save results
        combined_df.to_csv(os.path.join(output_dir, 'crema_d_features.csv'), index=False)
        features_df.to_csv(os.path.join(output_dir, 'raw_features.csv'), index=False)
        metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
        
        print(f"\nFeature extraction completed!")
        print(f"Total files processed: {len(all_features)}")
        print(f"Features extracted: {len(features_df.columns)}")
        print(f"Files saved to: {output_dir}")
        
        # Display summary
        print(f"\nDataset Summary:")
        print(f"Emotions: {combined_df['emotion'].value_counts().to_dict()}")
        print(f"Intensities: {combined_df['intensity'].value_counts().to_dict()}")
        print(f"Actors: {combined_df['actor_id'].nunique()}")
        
        # Display feature statistics
        print(f"\nFeature Statistics:")
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Feature shape: {combined_df.shape}")
        
    else:
        print("No features extracted successfully!")

if __name__ == "__main__":
    main()
