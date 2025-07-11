rm(list=ls())
library(dplyr)
library(tidyr)
library(tidyverse)
library(cluster)
library(factoextra)
library(FactoMineR)
library(corrplot)
library(RColorBrewer)
library(maps)
library(ggplot2)
library(plotly)

# Load gridded dataset
data_FRtraits_filled = read_csv("FR_alltraits_sites_filled_BHPMF.csv")
data_Griddedtraits = read.csv("dataset/site_gridded_filled.csv") %>%
  rename(SiteLatitude = site_lat, SiteLongitude = site_lon)

data_merged = data_FRtraits_filled %>%
  left_join(data_Griddedtraits, by = c("SiteLatitude", "SiteLongitude"))

colnames(data_merged)
write_csv(data_merged, "FR_alltraits_sites_merged.csv")
# [1] "SiteLatitude"    "SiteLongitude"   "CWM_Rr25_Tj2001" "CWM_RD"          "CWM_SRL"         "CWM_RTD"         "CWM_Exudation"  
# [8] "root_dep.value"  "olp.value"       "z_cost.value"    "ph.value"        "cn.value" 

# ====== DATA PREPARATION FOR CLUSTERING ======

# Remove rows with missing values and prepare clustering dataset
data_clean = data_merged %>%
  drop_na() %>%
  filter_if(is.numeric, all_vars(!is.infinite(.)))

print(paste("Clean dataset has", nrow(data_clean), "complete observations"))

# Select variables for clustering (functional traits + key environmental variables)
clustering_vars = c("CWM_Rr25_Tj2001", "CWM_RD", "CWM_SRL", "CWM_RTD",
                   "CWM_Exudation", "root_dep.value", "z_cost.value", "ph.value", "cn.value", "olp.value")

# Extract clustering data
clustering_data = data_clean %>%
  select(all_of(clustering_vars))

# Check data summary
summary(clustering_data)

# ====== CORRELATION ANALYSIS ======

# Compute correlation matrix
cor_matrix = cor(clustering_data, use = "complete.obs")

# Visualize correlation matrix
pdf("correlation_matrix.pdf", width = 10, height = 8)
corrplot(cor_matrix, method = "color", type = "upper",
         order = "hclust", tl.cex = 0.8, tl.col = "black",
         title = "Correlation Matrix of Functional Traits and Environmental Variables",
         mar = c(0,0,1,0))
dev.off()

# ====== DATA STANDARDIZATION ======

# Standardize variables (z-score normalization)
clustering_data_scaled = scale(clustering_data)

# Convert back to dataframe
clustering_data_scaled = as.data.frame(clustering_data_scaled)

# ====== PRINCIPAL COMPONENT ANALYSIS ======

# Perform PCA
pca_result = PCA(clustering_data_scaled, graph = FALSE)

# PCA summary
print("PCA Summary:")
print(summary(pca_result))

# Visualize PCA
pdf("pca_analysis.pdf", width = 15, height = 10)
par(mfrow = c(2, 3))

# Scree plot
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50))

# Variables contribution to PC1 and PC2
fviz_contrib(pca_result, choice = "var", axes = 1, top = 10)
fviz_contrib(pca_result, choice = "var", axes = 2, top = 10)

# Biplot
fviz_pca_biplot(pca_result, repel = TRUE, col.var = "#2E9FDF",
                col.ind = "#696969", title = "PCA Biplot")

# Variables plot
fviz_pca_var(pca_result, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))

dev.off()

# ====== DETERMINE OPTIMAL NUMBER OF CLUSTERS ======

# Elbow method for K-means
set.seed(123)
elbow_result = fviz_nbclust(clustering_data_scaled, kmeans, method = "wss", k.max = 10)

# Silhouette method
silhouette_result = fviz_nbclust(clustering_data_scaled, kmeans, method = "silhouette", k.max = 10)

# Gap statistic
gap_result = fviz_nbclust(clustering_data_scaled, kmeans, method = "gap_stat", k.max = 10)

# Save cluster validation plots
pdf("cluster_validation.pdf", width = 15, height = 5)
par(mfrow = c(1, 3))
print(elbow_result)
print(silhouette_result)
print(gap_result)
dev.off()

# ====== K-MEANS CLUSTERING ======

# Based on validation methods, try different k values
k_optimal = 4  # Adjust based on validation results

set.seed(123)
kmeans_result = kmeans(clustering_data_scaled, centers = k_optimal, nstart = 25)

# Add cluster assignments to original data
data_clustered = data_clean %>%
  mutate(
    Cluster_KMeans = as.factor(kmeans_result$cluster),
    PC1 = pca_result$ind$coord[,1],
    PC2 = pca_result$ind$coord[,2],
    PC3 = pca_result$ind$coord[,3]
  )

# Cluster summary
print("K-Means Clustering Summary:")
print(table(data_clustered$Cluster_KMeans))

# ====== HIERARCHICAL CLUSTERING ======

# Compute distance matrix
dist_matrix = dist(clustering_data_scaled, method = "euclidean")

# Hierarchical clustering
hclust_result = hclust(dist_matrix, method = "ward.D2")

# Cut dendrogram to get clusters
h_clusters = cutree(hclust_result, k = k_optimal)

# Add hierarchical cluster assignments
data_clustered$Cluster_Hierarchical = as.factor(h_clusters)

# Plot dendrogram
pdf("dendrogram.pdf", width = 12, height = 8)
fviz_dend(hclust_result, k = k_optimal, cex = 0.5, k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"))
dev.off()

# ====== CLUSTER CHARACTERIZATION ======

# Calculate cluster means for each variable
cluster_means = data_clustered %>%
  group_by(Cluster_KMeans) %>%
  summarise(
    across(all_of(clustering_vars), mean, na.rm = TRUE),
    Count = n(),
    .groups = "drop"
  )

print("Cluster Characteristics (K-Means):")
print(cluster_means)

# Create heatmap of cluster characteristics
cluster_means_matrix = cluster_means %>%
  select(-Count) %>%
  column_to_rownames("Cluster_KMeans") %>%
  as.matrix()

# Standardize for heatmap
cluster_means_scaled = scale(cluster_means_matrix)

pdf("cluster_heatmap.pdf", width = 10, height = 6)
heatmap(cluster_means_scaled,
        col = colorRampPalette(c("blue", "white", "red"))(50),
        main = "Cluster Characteristics Heatmap",
        xlab = "Variables", ylab = "Clusters")
dev.off()

# ====== VISUALIZATIONS ======

# 1. PCA plot colored by clusters
pca_cluster_plot = ggplot(data_clustered, aes(x = PC1, y = PC2, color = Cluster_KMeans)) +
  geom_point(size = 2, alpha = 0.7) +
  stat_ellipse(level = 0.68, type = "norm") +
  scale_color_brewer(type = "qual", palette = "Set1") +
  labs(
    title = "K-Means Clusters in Reduced Space",
    x = paste0("PC1 (", round(pca_result$eig[1,2], 1), "% variance)"),
    y = paste0("PC2 (", round(pca_result$eig[2,2], 1), "% variance)"),
    color = "Strategy Type"
  ) +
  theme_minimal() +
  scale_x_continuous(expand = c(-1, .8)) +
  theme(legend.position = "bottom")

ggsave("scatter_clusters.pdf", pca_cluster_plot, width = 5, height = 5)

# 2. Global map with clusters
world_map = map_data("world")

map_plot = ggplot() +
  geom_polygon(data = world_map, aes(x = long, y = lat, group = group),
               fill = "lightgray", color = "white", size = 0.1) +
  geom_point(data = data_clustered,
             aes(x = SiteLongitude, y = SiteLatitude, color = Cluster_KMeans),
             size = 1.5, alpha = 0.8) +
  scale_color_brewer(type = "qual", palette = "Set1") +
  labs(
    title = "Distribution of trait strategic clusters related to Z_cost",
    x = "Longitude", y = "Latitude",
    color = "Strategy Type"
  ) +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 8),
    legend.position = "bottom"
  ) +
  coord_fixed(ratio = 1.3)

ggsave("global_clusters_map.pdf", map_plot, width = 8, height = 4.5)

# 3. Z_cost vs other key traits by cluster
z_cost_plot = ggplot(data_clustered, aes(x = z_cost.value, y = log(CWM_RD), color = Cluster_KMeans)) +
  geom_point(size = 2, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE) +
  scale_color_brewer(type = "qual", palette = "Set1") +
  labs(
    title = "Z_cost vs Specific Root Length by Strategy Type",
    x = "Z_cost",
    color = "Strategy Type") +
  theme_minimal() + facet_wrap(~Cluster_KMeans, nrow = 2)
ggsave("z_cost_RD_clusters.pdf", z_cost_plot, width = 8, height = 6, dpi = 300)

# 4. 3D PCA plot (interactive)
plot_3d = plot_ly(data_clustered,
                  x = ~PC1, y = ~PC2, z = ~PC3,
                  color = ~Cluster_KMeans,
                  colors = c("#E31A1C", "#1F78B4", "#33A02C", "#FF7F00"),
                  type = "scatter3d", mode = "markers",
                  marker = list(size = 3)) %>%
  layout(
    title = "3D PCA Plot of Functional Strategy Types",
    scene = list(
      xaxis = list(title = paste0("PC1 (", round(pca_result$eig[1,2], 1), "%)")),
      yaxis = list(title = paste0("PC2 (", round(pca_result$eig[2,2], 1), "%)")),
      zaxis = list(title = paste0("PC3 (", round(pca_result$eig[3,2], 1), "%)"))))

# Save 3D plot as HTML
htmlwidgets::saveWidget(plot_3d, "3d_pca_clusters.html")

# ====== CLUSTER INTERPRETATION ======

# Generate cluster interpretation report
cluster_interpretation = data_clustered %>%
  group_by(Cluster_KMeans) %>%
  summarise(
    n_sites = n(),
    mean_z_cost = round(mean(z_cost.value, na.rm = TRUE), 3),
    mean_SRL = round(mean(CWM_SRL, na.rm = TRUE), 3),
    mean_RTD = round(mean(CWM_RTD, na.rm = TRUE), 3),
    mean_exudation = round(mean(CWM_Exudation, na.rm = TRUE), 3),
    mean_root_depth = round(mean(root_dep.value, na.rm = TRUE), 3),
    mean_pH = round(mean(ph.value, na.rm = TRUE), 3),
    mean_CN = round(mean(cn.value, na.rm = TRUE), 3),
    latitude_range = paste(round(min(SiteLatitude), 1), "to", round(max(SiteLatitude), 1)),
    .groups = "drop"
  )

print("=== FUNCTIONAL STRATEGY TYPES INTERPRETATION ===")
print(cluster_interpretation)

# Save results
write_csv(data_clustered, "clustered_dataset.csv")
write_csv(cluster_interpretation, "cluster_interpretation.csv")

print("=== CLUSTERING ANALYSIS COMPLETE ===")
print("Files generated:")
print("- correlation_matrix.pdf: Correlation between variables")
print("- pca_analysis.pdf: PCA results and biplots")
print("- cluster_validation.pdf: Optimal cluster number analysis")
print("- dendrogram.pdf: Hierarchical clustering dendrogram")
print("- cluster_heatmap.pdf: Cluster characteristics heatmap")
print("- pca_clusters.png: PCA plot with cluster colors")
print("- global_clusters_map.png: World map showing strategy types")
print("- z_cost_srl_clusters.png: Z_cost vs SRL by cluster")
print("- 3d_pca_clusters.html: Interactive 3D PCA plot")
print("- clustered_dataset.csv: Full dataset with cluster assignments")
print("- cluster_interpretation.csv: Summary of cluster characteristics")
