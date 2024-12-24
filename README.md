# COSWARA Dataset Analysis and COVID-19 Prediction

This repository contains an in-depth analysis of the COSWARA dataset, including metadata and audio files, to understand respiratory illnesses and predict COVID-19 status. Statistical techniques and a machine learning model were employed to analyze and classify health conditions.

---

## Key Findings

1. **Impact of Age on Audio Features:**

   - Age significantly influences audio features like 'rms_cough_heavy', 'rms_cough_shallow', and 'zero_crossing_rate_counting_fast'.
   - Distinct patterns in cough intensity and frequency were observed across different age groups.

2. **Role of Gender, Age, and Health Status:**

   - Gender, health status, and age interactions significantly impact audio characteristics, particularly 'rms_cough_heavy' and 'rms_cough_shallow'.

3. **Relationship Between COVID-19 Symptoms:**

   - Strong associations exist between COVID-19-related symptoms (e.g., cough, fatigue, loss of smell) and the COVID status variable.

4. **RMS Feature Analysis for Cough Heavy Audio:**

   - Non-parametric tests showed no significant differences in the RMS feature for cough-heavy audio between COVID-positive and negative groups in the 18–40 age range.

5. **Model Accuracy:**
   - The CNN model trained on mel spectrograms from breathing-deep audio achieved a validation accuracy of 72.9%.

---

## Objective

1. Conduct exploratory data analysis (EDA) on metadata to identify patterns, distributions, and anomalies.
2. Analyze how age, health status, and gender influence audio characteristics.
3. Use statistical methods (T-tests, Chi-Square, ANOVA) to find relationships within the dataset.
4. Develop a CNN model to classify health status using mel spectrograms.

---

## Data Description

- **Participants:** 2,745 individuals
- **Audio Files:** 24,705 recordings across nine types per participant.
- **Metadata Variables:** 41 demographic and health-related attributes.

---

## Methodology

### Metadata Cleaning

- Missing values were handled with logical assumptions.
- Data was filtered and grouped for relevant analysis.

### Exploratory Data Analysis

- Audio features like RMS and ZCR were analyzed and visualized.
- Groups were formed based on age, gender, and health status for meaningful comparisons.

### Statistical Analysis

- **Chi-Square Test:** Explored relationships between categorical variables.
- **T-Test and Mann Whitney Test:** Compared audio features across groups.
- **ANOVA:** Evaluated the effect of gender, age, and health status on audio features.

### Machine Learning

- **Model:** Convolutional Neural Network (CNN)
- **Input:** Mel spectrogram images extracted from audio files.
- **Performance:** Training accuracy of 81.54% and validation accuracy of 72.9%.

---

## Recommendations

1. Standardize audio recording conditions to improve data consistency.
2. Apply data preprocessing techniques like noise reduction and normalization.
3. Use additional audio types for better model accuracy.
4. Employ robust statistical tests for non-normal data distributions.

---

## Acknowledgments

This project is part of the DANA-4800 course at Langara College, and the dataset was sourced from the COSWARA team.

---

## References

- COSWARA Dataset Documentation
- Librosa Python Library Documentation
