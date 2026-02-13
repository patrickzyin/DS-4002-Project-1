# DS 4002 - Project 1

## Section 0: Project Overview
This repository contains an analysis of U.S. Presidential State of the Union addresses. The goal of this project is to evaluate whether the sentiment of these speeches differs during periods of national crisis compared to periods of stability. 

**Hypothesis:**
During times of national crisis, more than 50% of State of the Union addresses given in the same year as the crisis will be classified as more optimistic. 

**Research Question:** 
Will the sentiment score of State of the Union addresses be significantly lower during years classified as periods of national crisis compared to the average score during years of relative stability? 


## Section 1: Software and Platform 

**Software:** Python 3.8 or higher

**Required Packages:**
- `transformers` (>=4.30.0) - For sentiment analysis model
- `torch` (>=2.0.0) - Backend for transformers
- `pandas` (>=1.3.0) - Data manipulation and analysis
- `numpy` (>=1.21.0) - Numerical computing
- `matplotlib` (>=3.5.0) - Data visualization
- `seaborn` (>=0.12.0) - Statistical data visualization
- `scipy` (>=1.7.0) - Statistical tests (two-sample t-test)

**Installation:**
Can be installed with 
pip install -r requirements.txt

## Section 2: A Map of Documentation 

**DS-4002-Project-1/**
- README.md - Project overview and instructions
- requirements.txt - Python package dependencies

**DATA/**
- SOTU_data_final.csv - Dataset containing all SOTU speeches (1850-2025); Columns: President, Year, Title, Text, is_crisis, crisis_name
- state_of_the_union_texts.csv - Raw original dataset

**SCRIPTS/**
- sentiment_model.py - Main script: Analyzes sentiment of all speeches
- stat_analysis.py - Secondary script: Performs statistical analysis on results and creates visualizations

**OUTPUTS/**
- sentiment_analysis_results.csv - Results from sentiment analysis 
- analysis_summary.txt - Statistical test results summary 
- boxplot_positive_sentiment.png - Box plot comparing crisis vs non-crisis 
- histogram_positive_sentiment.png - Histogram of sentiment distribution 
- pie_crisis_sentiment.png - Pie chart for crisis speeches 
- pie_noncrisis_sentiment.png - Pie chart for non-crisis speeches 
- boxplot_ratio.png - Box plot of positive/negative ratios 
- timeseries_sentiment.png - Time series showing sentiment trends

## Section 3: Instructions for reproducing results


### Step 1: Set Up Environment

1. Clone or download this repository to your local machine
2. Open a terminal or command prompt and navigate to the project directory:
```bash
   cd DS-4002-Project-1
```
3. Install required Python packages:
```bash
   pip install -r requirements.txt
```

### Step 2: Verify Data Files

1. Ensure the `DATA` folder contains `SOTU_data_final.csv`
2. This file should contain 231 rows (one per speech) with the following columns:
   - President: Name of the president
   - Year: Year of the speech (1850-2025)
   - Title: Full title of the speech
   - Text: Complete speech text (stored as a list of sentences)
   - is_crisis: Binary indicator (1 = crisis, 0 = non-crisis)
   - crisis_name: Name of the crisis (if applicable)


### Step 3: Run Sentiment Analysis

1. Navigate to the SCRIPTS folder:
```bash
   cd SCRIPTS
```
2. Run the sentiment analysis script:
```bash
   python sentiment_model.py
```
3. This script will:
   - Load all 231 speeches from the CSV file
   - Clean the text (remove audience reactions like applause, laughter)
   - Analyze each sentence using the DistilBERT sentiment model
   - Classify each speech as POSITIVE (>50% positive sentences) or NEGATIVE (≤50%)
   - Save results to `../OUTPUTS/sentiment_analysis_results.csv`

4. Verify the output file was created in the OUTPUTS folder

### Step 4: Run Statistical Analysis

1. From the SCRIPTS folder, run the stat analysis script:
```bash
   python stat_analysis.py
```
2. This script will:
   - Load the sentiment analysis results
   - Separate speeches into crisis (n=82) and non-crisis (n=87) groups
   - Calculate descriptive statistics for both groups
   - Perform a two-sample t-test on the positive/negative ratios
   - Generate 6 visualizations comparing the two groups
   - Save a summary report with statistical test results

3. Check the OUTPUTS folder for all generated files




## Section 5: References

[1]	National Bureau of Economic Research, “US Business Cycle Expansions and Contractions,” nber.org. https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions (accessed Feb. 4, 2026).
[2]	Federal Reserve Bank of St. Louis, “Recession Bars,” FRED Help. https://fredhelp.stlouisfed.org/fred/data/understanding-the-data/recession-bars/ (accessed Feb. 4, 2026).
[3] 	U.S. Department of Veterans Affairs, “America’s Wars,” department.va.gov. https://department.va.gov/americas-wars/#world-war-ii-1941-1945 (accessed Feb. 4, 2026).
[4]	 D. J. Trump, “Remarks by President Trump in State of the Union Address,” Trump White House Archives, Feb. 4, 2020. [Online]. Available: https://trumpwhitehouse.archives.gov/briefings-statements/remarks-president-trump-state-union-address-3/ (accessed Feb. 4, 2026).
[5] 	J. R. Biden, “Read the full transcript of President Joe Biden's first speech to Congress,” USA Today, Apr. 28, 2021. [Online]. Available: https://www.usatoday.com/story/news/politics/2021/04/28/joe-bidens-speech-congress-read-full-transcript/4883244001/ (accessed Feb. 4, 2026).
[6] 	J. R. Biden, “Address Before a Joint Session of the Congress on the State of the Union,” The American Presidency Project, Mar. 1, 2022. [Online]. Available: https://www.presidency.ucsb.edu/documents/address-before-joint-session-the-congress-the-state-the-union-28 (accessed Feb. 4, 2026).
[7] 	J. R. Biden, “State of the Union 2023: Full Transcript of Biden’s Address,” The New York Times, Feb. 8, 2023. [Online]. Available: https://www.nytimes.com/2023/02/08/us/politics/biden-state-of-the-union-transcript.html (accessed Feb. 4, 2026).
[8] 	U.S. Congress, House Committee on the Judiciary, “Hearing on the Report of the Special Counsel on the Investigation Into Unauthorized Removal, Retention, and Possession of Classified and Other Confidential Documents and Materials,” 118th Cong., Mar. 12, 2024. [Online]. Available: https://www.congress.gov/118/meeting/house/116942/documents/HHRG-118-JU00-20240312-SD001.pdf (accessed Feb. 4, 2026).
[9] 	J. R. Biden, “Address Before a Joint Session of the Congress,” The American Presidency Project, Mar. 7, 2024. [Online]. Available: https://www.presidency.ucsb.edu/documents/address-before-joint-session-the-congress-4 (accessed Feb. 4, 2026).
