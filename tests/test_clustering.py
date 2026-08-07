import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from src.segmentation.clustering import scale_features, run_grid_search, map_personas

class TestClustering(unittest.TestCase):
    
    def setUp(self):
        # Create a mock customer behavior dataframe
        np.random.seed(42)
        n_samples = 100
        
        self.mock_df = pd.DataFrame({
            "customer_id": np.arange(1000, 1000 + n_samples, dtype=float),
            "recency": np.random.exponential(scale=100, size=n_samples),
            "frequency": np.random.poisson(lam=5, size=n_samples) + 1,
            # include negative spend to simulate returns
            "monetary_total": np.random.normal(loc=500, scale=1000, size=n_samples),
            "seasonal_concentration": np.random.uniform(0, 1, size=n_samples),
            "discount_sensitivity": np.random.uniform(0, 0.2, size=n_samples),
            "engagement_score": np.random.uniform(0.1, 0.9, size=n_samples)
        })
        # Ensure at least one wholesale-like extreme customer
        self.mock_df.loc[0, "monetary_total"] = 500000.0
        self.mock_df.loc[0, "frequency"] = 250
        
        self.features = ['recency', 'frequency', 'monetary_total', 'seasonal_concentration']
        
    def test_scale_features(self):
        X_scaled, pt = scale_features(self.mock_df, self.features)
        
        # Check output shape
        self.assertEqual(X_scaled.shape, (100, len(self.features)))
        self.assertIsInstance(pt, PowerTransformer)
        
        # Verify columns are standardized (approximately mean 0, variance 1)
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0)
        
        for i, col in enumerate(self.features):
            self.assertAlmostEqual(means[i], 0, delta=0.5, msg=f"{col} mean should be near 0")
            self.assertAlmostEqual(stds[i], 1, delta=0.5, msg=f"{col} std should be near 1")

    def test_run_grid_search(self):
        X_scaled, _ = scale_features(self.mock_df, self.features)
        results = run_grid_search(X_scaled, max_k=5)
        
        self.assertIn("k", results)
        self.assertIn("inertia", results)
        self.assertIn("silhouette", results)
        
        # Check evaluated values
        self.assertEqual(results["k"], [2, 3, 4, 5])
        self.assertEqual(len(results["inertia"]), 4)
        self.assertEqual(len(results["silhouette"]), 4)
        
        # Silhouette scores must be between -1 and 1
        for score in results["silhouette"]:
            self.assertTrue(-1 <= score <= 1)

    def test_map_personas(self):
        # Create mock centroids representing 4 segments
        centroids_df = pd.DataFrame([
            # Cluster 0: Long lapsed, low frequency, high seasonal
            {"mean_recency": 320.0, "mean_seasonal": 0.85, "mean_monetary": 450.0},
            # Cluster 1: Long lapsed, low frequency, low seasonal
            {"mean_recency": 290.0, "mean_seasonal": 0.02, "mean_monetary": 300.0},
            # Cluster 2: Active, high frequency, high monetary
            {"mean_recency": 45.0, "mean_seasonal": 0.20, "mean_monetary": 4500.0},
            # Cluster 3: Wholesale outlier
            {"mean_recency": 5.0, "mean_seasonal": 0.15, "mean_monetary": 120000.0}
        ])
        
        persona_map = map_personas(centroids_df)
        
        # Check mapping outputs
        self.assertEqual(len(persona_map), 4)
        self.assertEqual(persona_map[3], "Wholesale / VIP Buyers")
        self.assertEqual(persona_map[0], "Seasonal Gift Shoppers")
        self.assertEqual(persona_map[2], "Active Loyal Customers")
        self.assertEqual(persona_map[1], "Lapsed Low-Value Customers")

if __name__ == "__main__":
    unittest.main()
