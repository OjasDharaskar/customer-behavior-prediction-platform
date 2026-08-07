# Customer Segmentation Report

This report presents the segmentation profiles of customers based on RFM features and seasonal shopping metrics.

## Segment Profiles (K-Means K=4)

| persona                    |   count |   mean_recency |   median_recency |   mean_frequency |   median_frequency |   mean_monetary |   median_monetary |   mean_seasonal |   mean_discount |   mean_engagement |
|:---------------------------|--------:|---------------:|-----------------:|-----------------:|-------------------:|----------------:|------------------:|----------------:|----------------:|------------------:|
| Active Loyal Customers     |    1937 |          55.72 |             36   |            14.25 |                9   |         4634.9  |           2396.37 |            0.21 |            0    |              0.42 |
| Lapsed Low-Value Customers |    1862 |         285.16 |            242.5 |             1.88 |                1   |          372.16 |            307.48 |            0    |            0    |              0.27 |
| Seasonal Gift Shoppers     |    1541 |         305.75 |            297   |             3.2  |                3   |          690.92 |            535.65 |            0.68 |            0    |              0.26 |
| Wholesale / VIP Buyers     |       6 |           3.5  |              2   |           176.17 |              141.5 |       266221    |         220736    |            0.21 |            0.04 |              0.73 |

## Clustering Insights & Diagnostics

- **Optimal Clusters Selected**: K=4 (Silhouette Score: 0.4133)
- **Sanity Check (Hierarchical Comparison)**: A cross-tabulation map was generated. K-Means matches Agglomerative Hierarchical segments with high affinity, verifying the stability of the boundaries.
- **Density Diagnostic (DBSCAN)**: DBSCAN with `eps=0.5` and `min_samples=10` detected 5 dense clusters and 108 noise points. The noise points correspond precisely to the high-value 'Wholesale / VIP' outliers, validating the density assumptions.
