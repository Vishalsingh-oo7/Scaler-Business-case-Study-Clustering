## Scaler-Business-case-Study-Clustering

**Problem Statement**

> **Scaler is an online tech-versity offering intensive computer science & Data Science courses through live classes delivered by tech leaders and subject matter experts. The meticulously structured program enhances the skills of software professionals by offering a modern curriculum with exposure to the latest technologies. It is a product by InterviewBit.**

>> **Working as a data scientist with the analytics vertical of Scaler, focused on profiling the best companies and job positions to work for from the Scaler database. Information provided for a segment of learners and tasked to cluster them on the basis of their job profile, company, and other features. Ideally, these clusters should have similar characteristics**.


**Data Dictionary:**

* `Unnamed 0` - Index of the dataset
* `Email_hash` - Anonymised Personal Identifiable Information (PII)
* `Company_hash` - This represents an anonymized identifier for the company, which is the current employer of the learner.
* `orgyear` - Employment start date
* `CTC` - Current CTC
* `Job_position` - Job profile in the company
* `CTC_updated_year` - Year in which CTC got updated (Yearly increments, Promotions)

## Methodology

### Preprocessing

1.  **Removing Duplicates and Handling PII**: Identified and addressed duplicate entries related to `email_hash` where a single PII had multiple entries. `email_hash` and `Unnamed: 0` columns were dropped to avoid confusion and redundant data.
2.  **Text Cleaning for Categorical Features**: Applied a `remove_special` function to `company_hash` and `job_position` columns to remove special characters and digits, followed by lowercasing and stripping whitespace. This standardizes the text data.
3.  **Dropping Empty Entries**: Removed rows where `company_hash` or `job_position` became empty after text cleaning.
4.  **Handling Null Values in `orgyear`**: Null values in `orgyear` were imputed using the median `orgyear` grouped by `company_hash` (Mean Target Imputation). Remaining nulls were dropped.
5.  **Handling Null Values in `job_position`**: Replaced 'nan' string values in `job_position` with `np.nan`, and then filled any resulting actual NaN values with 'Others' to ensure consistency.
6.  **Masking Low-Frequency Company Hashes**: Companies with 5 or fewer occurrences were grouped into an 'Others' category to reduce cardinality and focus on more prominent entities.
7.  **Final Duplicate Removal**: A final check and removal of any remaining duplicate rows after all preprocessing steps.

### Feature Engineering

1.  **Years of Experience Calculation**: A new column `years_of_experience` was created by subtracting the `orgyear` from the current year (2022). The `orgyear` was capped at 2022 to handle any future or erroneous entries.
   
 ```  
data['years_of_experience'] = 2022 - data['orgyear']
```

2.  **CTC Updated Year Adjustment**: The `ctc_updated_year` was adjusted to ensure it is not before the `orgyear`. If `ctc_updated_year` was less than `orgyear`, it was set to `orgyear`.
   
```
data['ctc_updated_year'] = data.apply(lambda x: x['orgyear'] if x['ctc_updated_year'] < x['orgyear'] else x['ctc_updated_year'], axis=1)`
```
    
3.  **Manual Clustering Features**: Additional features like `designation` and `classs` were created based on `ctc` percentiles within grouped `years_of_experience`, `job_position`, and `company_hash` combinations. Similar `tier` and `company_cluster_n`, `company_job_cluster_n` features were created using IQR-based clustering logic on median CTC values to categorize companies and job positions.


### Exploratory Data Analysis (EDA)

1.  **Univariate Analysis**: Explored individual features to understand their distributions and characteristics.
    *   **Categorical Features (`company_hash`, `job_position`, `orgyear`, `ctc_updated_year`)**: Bar plots were used to visualize the top 15 most frequent categories for each, providing insights into the most common companies, job positions, and years of employment/CTC updates.
    *   **Continuous Features (`ctc`)**: A distribution plot (histogram with KDE) and a box plot were used to examine the distribution of `ctc` values, revealing its range and potential outliers. It was noted that `ctc` had a large range, prompting consideration for scaling during visualization.

2.  **Multivariate Analysis**: Examined relationships between multiple features.
    *   **Top 50 Highest Paying Job Positions**: A bar plot generated using Plotly was used to visualize the top 50 job positions based on their maximum `ctc`. This helped identify roles that command the highest salaries in the dataset.


## Unsupervised Learning - Clustering

To group learners based on their job profiles, company, and other features, two unsupervised learning techniques were employed: K-Means Clustering and Hierarchical Clustering. Before applying these algorithms, several data preparation steps were crucial.

### Data Preparation for Clustering:

1.  **Feature Selection**: Features relevant for clustering were identified.
2.  **Label Encoding**: Categorical features like `company_hash` and `job_position` were converted into numerical representations using `LabelEncoder` from `sklearn.preprocessing`. This is essential because clustering algorithms require numerical input.
3.  **Feature Scaling**: The `ctc` (Current CTC) column, a numerical feature, was scaled using `MinMaxScaler` from `sklearn.preprocessing`. This ensures that features with larger values do not disproportionately influence the distance calculations in the clustering algorithms.
4.  **Column Dropping**: Columns `orgyear` and `ctc_updated_year` were dropped from the dataset used for clustering (`x`) as they were either already transformed into `years_of_experience` or deemed less relevant for direct clustering after other features were engineered.

### K-Means Clustering:

*   **Elbow Method**: The Elbow Method was used to determine the optimal number of clusters (K). This method involves plotting the sum of squared distances of samples to their closest cluster center (inertia) against the number of clusters. The "elbow point" in the plot, where the rate of decrease in inertia sharply changes, suggests the optimal K. In this analysis, a K-value of **3** was observed as the optimal number of clusters.
*   **K-Means Application**: After identifying K=3, the `KMeans` algorithm was applied to the prepared dataset. Each data point was then assigned a cluster label (`k-m label`), indicating which of the three clusters it belongs to.

### Hierarchical Clustering:

*   **Sampling for Performance**: Due to the computational intensity of Hierarchical Clustering, especially with large datasets, a small sample of the data (0.25%, `z = x.sample(frac=0.0025)`) was taken for this analysis.
*   **Dendrogram Visualization**: A dendrogram was generated using `scipy.cluster.hierarchy.dendrogram` to visually represent the hierarchical relationships between data points. This visualization aids in understanding the structure of the data and can also help in choosing the number of clusters by looking for distinct branches.
*   **Agglomerative Clustering**: The `AgglomerativeClustering` model from `sklearn.cluster` was then applied to the sampled data with `n_clusters=3` (consistent with the K-Means result) and `linkage='ward'`. This assigned a hierarchical cluster label (`Aglo-label`) to each data point in the sample.

These steps provided the foundation for analyzing different groups of learners based on their professional attributes.


## ✨ Insights

- **Top paying job titles** across clusters include:  
  *Engineering Leadership, Backend Engineer, FullStack Engineer, Android Engineer, Data Scientist, SDET, QA Engineer, Product Manager, Program Manager*

- **Job titles showing declining average salary in recent years** include:  
  *QA Engineer, System Administrator, Support Engineer*

- **Job roles with consistent average salary increase over years** include:  
  *Backend Engineer, SDET, FullStack Engineer, Data Scientist, Android Engineer*

- **Roles with strong early-career salary growth** include:  
  *QA Engineer, SDET, Support Engineer, Android Engineer*

- **Job roles with niche or specialized demand clusters** include:  
  *Android Engineer, Machine Learning Engineer, DevOps Engineer, Product Manager, Data Engineer*

- **Clusters reveal distinct groups for technical vs. managerial job families**, with:  
  - *Product Manager and Program Manager roles generally forming higher-salary clusters*  
  - *QA and Support roles forming moderate-salary clusters*

- **Average salaries across most roles show slight downward trends year over year**, suggesting:  
  *a potential industry-wide normalization or saturation in certain skill areas*

- **Roles focused on backend, data, and mobile development** appear in clusters with:  
  *higher silhouette scores and clearer separation — indicating better-defined salary differentiation*


## 📌 Recommendations

- **Freshers aiming for high long-term salary growth** should target roles like:  
  *Backend Engineer, Android Engineer, SDET, QA Engineer, Data Scientist, FullStack Developer*

- **Those seeking early-career growth with relatively fast salary jumps** should focus on:  
  *Support Engineer, QA Engineer, SDET*

- **Job titles consistently associated with strong compensation growth across years** include:  
  *Backend Engineer, Data Scientist, Android Engineer*

- **Candidates aiming for strategic or leadership roles** should consider long-term paths into:  
  *Product Manager, Program Manager, Engineering Leadership*

- **Avoid roles where average salaries are declining unless backed by strong niche skills or interest**, such as:  
  *System Administrator, QA Engineer (in low-growth environments)*

- **Focus on upskilling in emerging or in-demand areas** like:  
  *Machine Learning, DevOps, Cloud Engineering, Mobile App Development (Android/iOS)*

- **Roles with multi-domain skills (e.g., FullStack, Data Engineer)** offer wider opportunities and better salary stability over time.
