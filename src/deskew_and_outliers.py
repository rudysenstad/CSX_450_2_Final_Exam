numeric_df = housing_df.select_dtypes(exclude=['category']).copy()
numeric_df.drop("SalePrice", axis=1, inplace=True)
numeric_sc_df = (numeric_df - numeric_df.mean())/numeric_df.std()
numeric_log_df = np.log(numeric_df + 1)
numeric_log_sc_df = (numeric_df - numeric_df.mean())/numeric_df.std()

outliers = [198, 524, 1174, 1183, 1299, 186, 692, 770, 
            179, 225, 804, 889, 1387, 497]
numeric_final_df = numeric_log_sc_df.drop(outliers, axis=0)
housing_ouliers_removed_df = housing_df.drop(outliers, axis=0)