import streamlit as st
import pandas as pd
import numpy as np
from thefuzz import fuzz
from collections import defaultdict

# Helper function for CSV conversion with caching
def convert_df_to_csv(df):
    """Convert DataFrame to CSV bytes for download"""
    return df.to_csv(index=False).encode('utf-8')

# Set page configuration
st.set_page_config(
    page_title="Clarity - Data Quality Assistant",
    page_icon="✨",
    layout="wide"
)

# Main title
st.title("Clarity ✨")

# Introduction
st.write("""
Welcome to Clarity, your interactive data quality auditing assistant! 
Upload your CSV file below to begin analyzing and improving your data quality.
""")

# Section 1: Upload & Profile Your Data
st.header("1. Upload & Profile Your Data")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a CSV file to begin your data quality audit.",
    type=['csv']
)

# Conditional logic for the uploader
if uploaded_file is None:
    # If a dataframe is already in session state, but the uploader is empty,
    # it means the user has removed the file. Clear the state.
    if 'df' in st.session_state:
        del st.session_state['df']
        if 'duplicates' in st.session_state:
            del st.session_state['duplicates']
        if 'merge_choices' in st.session_state:
            del st.session_state['merge_choices']
        if 'file_name' in st.session_state:
            del st.session_state['file_name']
        st.rerun() # Rerun to update the UI to the 'no file' state
    
    st.info("💡 **Tip:** Upload a CSV file to get started with your data quality audit.")
else:
    # Only load the dataframe if it's not already in the session state
    # or if a new file has been uploaded.
    if 'df' not in st.session_state or st.session_state.get('file_name') != uploaded_file.name:
        try:
            # Load Data with Error Handling
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded '{uploaded_file.name}'.")
            
            # Store DataFrame and filename in Session State
            st.session_state['df'] = df
            st.session_state['file_name'] = uploaded_file.name # Keep track of the file name

            # Clear out old analysis results when a new file is loaded
            if 'duplicates' in st.session_state:
                del st.session_state['duplicates']
            if 'merge_choices' in st.session_state:
                del st.session_state['merge_choices']
            
            # Rerun to ensure the rest of the app sees the new df
            st.rerun()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            # If the dataframe was partially loaded, remove it to avoid inconsistent state
            if 'df' in st.session_state:
                del st.session_state['df']
            st.stop()

# The rest of the app logic now depends on 'df' being in session_state,
# but is un-indented from the file uploader logic.
if 'df' in st.session_state:
    # Create an Expandable Data Preview
    with st.expander("▶️ Expand to see a preview of your uploaded data"):
        st.dataframe(st.session_state.df.head())
    
    # Display the Automated Data Profile
    st.subheader("Automated Data Profile")
    
    # Use columns for side-by-side layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Number of rows:** {st.session_state.df.shape[0]}")
        st.write(f"**Number of columns:** {st.session_state.df.shape[1]}")
        st.write(f"**Duplicate rows:** {st.session_state.df.duplicated().sum()}")
        st.write(f"**Missing values:** {st.session_state.df.isnull().sum().sum()}")
    
    with col2:
        st.write("**Column Data Types:**")
        # Convert dtypes Series to DataFrame for better display - fix Arrow serialization issue
        dtypes_df = pd.DataFrame({
            'Column': st.session_state.df.dtypes.index.tolist(),
            'Data Type': [str(dtype) for dtype in st.session_state.df.dtypes.values]
        })
        st.dataframe(dtypes_df, use_container_width=True)
    
    # Display Detailed Statistics
    st.markdown("##### Descriptive Statistics (Numerical Columns)")
    st.dataframe(st.session_state.df.describe())
    
    st.markdown("##### Value Counts (Categorical Columns)")
    
    # Identify categorical columns
    categorical_columns = st.session_state.df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        selected_categorical = st.selectbox(
            "Select a categorical column to view value counts:",
            categorical_columns
        )
        
        if selected_categorical:
            value_counts = st.session_state.df[selected_categorical].value_counts()
            value_counts_df = pd.DataFrame({
                'Value': value_counts.index,
                'Count': value_counts.values
            })
            st.dataframe(value_counts_df, use_container_width=True)
    else:
        st.info("No categorical columns found in the dataset.")
    
    # Advanced Data Quality Report Section
    st.subheader("Advanced Data Quality Report")
    
    if st.button("Generate Detailed Quality Report"):
        with st.spinner("Generating comprehensive data quality report..."):
            try:
                # Custom data quality tests
                results_data = []
                total_tests = 0
                passed_tests = 0
                
                for column in st.session_state.df.columns:
                    col_data = st.session_state.df[column]
                    col_dtype = col_data.dtype
                    
                    # Test 1: Null values check
                    total_tests += 1
                    null_count = col_data.isnull().sum()
                    null_percentage = (null_count / len(col_data)) * 100
                    
                    if null_count == 0:
                        passed_tests += 1
                        status = "✅ Pass"
                        details = "No null values found"
                    else:
                        status = "❌ Fail"
                        details = f"Found {null_count} null values ({null_percentage:.2f}%)"
                    
                    results_data.append({
                        'Test Type': 'Null Values Check',
                        'Column': column,
                        'Status': status,
                        'Details': details
                    })
                    
                    # Test 2: Data type specific tests
                    if col_dtype in ['int64', 'float64']:
                        total_tests += 1
                        
                        # Check for outliers using IQR method
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        outlier_count = len(outliers)
                        outlier_percentage = (outlier_count / len(col_data)) * 100
                        
                        if outlier_count == 0:
                            passed_tests += 1
                            status = "✅ Pass"
                            details = "No outliers detected"
                        elif outlier_percentage < 5:  # Less than 5% outliers is acceptable
                            passed_tests += 1
                            status = "⚠️ Warning"
                            details = f"Found {outlier_count} outliers ({outlier_percentage:.2f}%)"
                        else:
                            status = "❌ Fail"
                            details = f"High outlier count: {outlier_count} ({outlier_percentage:.2f}%)"
                        
                        results_data.append({
                            'Test Type': 'Outlier Detection',
                            'Column': column,
                            'Status': status,
                            'Details': details
                        })
                        
                        # Test 3: Range consistency
                        total_tests += 1
                        range_value = col_data.max() - col_data.min()
                        if range_value > 0:
                            passed_tests += 1
                            status = "✅ Pass"
                            details = f"Range: {col_data.min():.2f} to {col_data.max():.2f}"
                        else:
                            status = "❌ Fail"
                            details = "No variation in values"
                        
                        results_data.append({
                            'Test Type': 'Range Consistency',
                            'Column': column,
                            'Status': status,
                            'Details': details
                        })
                        
                    elif col_dtype == 'object':
                        total_tests += 1
                        
                        # Check for unique value distribution
                        unique_count = col_data.nunique()
                        total_count = len(col_data)
                        unique_percentage = (unique_count / total_count) * 100
                        
                        if unique_count == 1:
                            status = "❌ Fail"
                            details = "All values are identical"
                        elif unique_percentage > 90:
                            status = "⚠️ Warning"
                            details = f"High cardinality: {unique_count} unique values ({unique_percentage:.2f}%)"
                        else:
                            passed_tests += 1
                            status = "✅ Pass"
                            details = f"Good distribution: {unique_count} unique values ({unique_percentage:.2f}%)"
                        
                        results_data.append({
                            'Test Type': 'Value Distribution',
                            'Column': column,
                            'Status': status,
                            'Details': details
                        })
                        
                        # Test 4: String length consistency (for text columns)
                        if col_data.str.len().std() > 0:
                            total_tests += 1
                            avg_length = col_data.str.len().mean()
                            length_std = col_data.str.len().std()
                            
                            if length_std < avg_length * 0.5:  # Standard deviation less than 50% of mean
                                passed_tests += 1
                                status = "✅ Pass"
                                details = f"Consistent length: avg={avg_length:.1f}, std={length_std:.1f}"
                            else:
                                status = "⚠️ Warning"
                                details = f"Variable length: avg={avg_length:.1f}, std={length_std:.1f}"
                            
                            results_data.append({
                                'Test Type': 'String Length Consistency',
                                'Column': column,
                                'Status': status,
                                'Details': details
                            })
                
                # Test 5: Overall dataset quality
                total_tests += 1
                duplicate_rows = st.session_state.df.duplicated().sum()
                duplicate_percentage = (duplicate_rows / len(st.session_state.df)) * 100
                
                if duplicate_rows == 0:
                    passed_tests += 1
                    status = "✅ Pass"
                    details = "No duplicate rows found"
                elif duplicate_percentage < 5:
                    status = "⚠️ Warning"
                    details = f"Found {duplicate_rows} duplicate rows ({duplicate_percentage:.2f}%)"
                else:
                    status = "❌ Fail"
                    details = f"High duplicate count: {duplicate_rows} rows ({duplicate_percentage:.2f}%)"
                
                results_data.append({
                    'Test Type': 'Duplicate Row Check',
                    'Column': 'Dataset',
                    'Status': status,
                    'Details': details
                })
                
                failed_tests = total_tests - passed_tests
                
                # Display Summary Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Tests", total_tests)
                with col2:
                    st.metric("✅ Passed", passed_tests)
                with col3:
                    st.metric("❌ Failed", failed_tests)
                
                # Display Detailed Results
                st.markdown("##### Detailed Test Results")
                
                # Create DataFrame and display
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Quality Score
                quality_score = (passed_tests / total_tests) * 100
                st.markdown(f"**Overall Data Quality Score: {quality_score:.1f}%**")
                
                if quality_score >= 90:
                    st.success("🎉 Excellent data quality!")
                elif quality_score >= 75:
                    st.warning("⚠️ Good data quality with some issues to address")
                else:
                    st.error("🚨 Poor data quality - significant issues detected")
                
            except Exception as e:
                st.error(f"Error generating quality report: {str(e)}")
                st.exception(e)

# Sidebar Cleaning Workbench - Only show if DataFrame is loaded
if 'df' in st.session_state:
    # Create the Sidebar Workbench
    st.sidebar.title("🛠️ Cleaning Workbench")
    st.sidebar.header("Handle Missing Values")
    
    # Test message to verify sidebar is working
    st.sidebar.info("✅ Sidebar is working! DataFrame loaded successfully.")
    
    # Debug information
    with st.sidebar.expander("🔍 Debug Info"):
        st.write(f"DataFrame shape: {st.session_state.df.shape}")
        st.write(f"Total missing values: {st.session_state.df.isnull().sum().sum()}")
        st.write("Missing values by column:")
        missing_by_col = st.session_state.df.isnull().sum()
        for col, count in missing_by_col.items():
            if count > 0:
                st.write(f"  {col}: {count}")
    
    # Automatically Detect Columns with Missing Values
    columns_with_missing = st.session_state.df.columns[st.session_state.df.isnull().any()].tolist()
    
    if not columns_with_missing:
        st.sidebar.success("✅ No missing values found in your dataset!")
    else:
        st.sidebar.info(f"Found {len(columns_with_missing)} columns with missing values: {', '.join(columns_with_missing)}")
        
        # Create Interactive Cleaning Controls
        selected_column = st.sidebar.selectbox(
            "Select a column to clean:",
            columns_with_missing
        )
        
        # Get data type of selected column
        is_numeric = pd.api.types.is_numeric_dtype(st.session_state.df[selected_column])
        
        # Create base strategies
        strategies = [
            "Drop rows with missing values",
            "Fill with mode",
            "Fill with a custom value"
        ]
        
        # Add numeric-specific strategies
        if is_numeric:
            strategies.extend([
                "Fill with mean",
                "Fill with median"
            ])
        
        # Strategy Selector
        selected_strategy = st.sidebar.radio(
            "Choose a cleaning strategy:",
            strategies
        )
        
        # Custom Value Input
        custom_value = None
        if selected_strategy == "Fill with a custom value":
            custom_value = st.sidebar.text_input(
                "Enter custom value:",
                placeholder="Type your value here..."
            )
        
        # Apply Button
        if st.sidebar.button("Apply Cleaning"):
            try:
                # Create a copy of the DataFrame
                df_cleaned = st.session_state.df.copy()
                
                # Store the original missing count for this column
                original_missing_count = st.session_state.df[selected_column].isnull().sum()
                
                st.sidebar.info(f"Processing: {original_missing_count} missing values in '{selected_column}'")
                
                # Implement the Backend Cleaning Logic
                if selected_strategy == "Drop rows with missing values":
                    df_cleaned = df_cleaned.dropna(subset=[selected_column])
                    st.sidebar.success(f"✅ Dropped {original_missing_count} rows with missing values in '{selected_column}'")
                
                elif selected_strategy == "Fill with mean":
                    mean_value = df_cleaned[selected_column].mean()
                    df_cleaned[selected_column] = df_cleaned[selected_column].fillna(mean_value)
                    st.sidebar.success(f"✅ Filled {original_missing_count} missing values in '{selected_column}' with mean: {mean_value:.2f}")
                
                elif selected_strategy == "Fill with median":
                    median_value = df_cleaned[selected_column].median()
                    df_cleaned[selected_column] = df_cleaned[selected_column].fillna(median_value)
                    st.sidebar.success(f"✅ Filled {original_missing_count} missing values in '{selected_column}' with median: {median_value:.2f}")
                
                elif selected_strategy == "Fill with mode":
                    mode_value = df_cleaned[selected_column].mode().iloc[0] if not df_cleaned[selected_column].mode().empty else "Unknown"
                    df_cleaned[selected_column] = df_cleaned[selected_column].fillna(mode_value)
                    st.sidebar.success(f"✅ Filled {original_missing_count} missing values in '{selected_column}' with mode: {mode_value}")
                
                elif selected_strategy == "Fill with a custom value":
                    if custom_value is not None and custom_value.strip() != "":
                        # Try to convert to numeric if the column is numeric
                        if is_numeric:
                            try:
                                custom_value = float(custom_value)
                            except ValueError:
                                st.sidebar.error("❌ Please enter a valid number for numeric columns")
                                st.stop()
                        
                        df_cleaned[selected_column] = df_cleaned[selected_column].fillna(custom_value)
                        st.sidebar.success(f"✅ Filled {original_missing_count} missing values in '{selected_column}' with: {custom_value}")
                    else:
                        st.sidebar.error("❌ Please enter a custom value")
                        st.stop()
                
                # Update the DataFrame in session state
                st.session_state.df = df_cleaned
                
                st.sidebar.success(f"✅ Updated DataFrame shape: {df_cleaned.shape}")
                
                # Rerun the app to refresh all displays
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"❌ Error during cleaning: {str(e)}")
                st.sidebar.exception(e)
else:
    st.sidebar.info("📁 Please upload a CSV file to access the cleaning workbench.")

# Section 2: Clean & Validate Data
st.header("2. Clean & Validate Data")

# Only show duplicate detection if DataFrame is loaded
if 'df' in st.session_state:
    # Duplicate Detection Section
    st.subheader("Detect Potential Duplicates")
    
    # Column Selector
    selected_columns = st.multiselect(
        "Select columns to check for duplicates:",
        options=st.session_state.df.columns.tolist(),
        default=st.session_state.df.columns.tolist()
    )
    
    # Threshold Slider
    similarity_threshold = st.slider(
        "Select the similarity threshold (%) for considering records as duplicates:",
        min_value=0,
        max_value=100,
        value=90,
        help="Higher values mean records must be more similar to be considered duplicates"
    )
    
    # Trigger Button
    if st.button("Find Potential Duplicates"):
        with st.spinner("Finding duplicates..."):
            try:
                # Create a temporary copy of the DataFrame
                temp_df = st.session_state.df.copy()
                
                # Consolidate data for comparison
                # Combine selected columns into a single string for each row
                combined_strings = temp_df[selected_columns].fillna('').astype(str).agg(' '.join, axis=1)
                
                # Initialize variables for duplicate detection
                potential_duplicates = defaultdict(list)
                processed_indices = set()
                
                # Implement the matching algorithm
                for i in range(len(combined_strings)):
                    if i in processed_indices:
                        continue
                    
                    current_cluster = [i]
                    processed_indices.add(i)
                    
                    for j in range(i + 1, len(combined_strings)):
                        if j in processed_indices:
                            continue
                        
                        # Calculate similarity score
                        similarity_score = fuzz.ratio(combined_strings.iloc[i], combined_strings.iloc[j])
                        
                        # If similarity is above threshold, add to cluster
                        if similarity_score >= similarity_threshold:
                            current_cluster.append(j)
                            processed_indices.add(j)
                    
                    # Only add clusters with more than one record
                    if len(current_cluster) > 1:
                        potential_duplicates[f"cluster_{len(potential_duplicates)}"] = current_cluster
                
                # Store results in session state
                st.session_state['duplicates'] = list(potential_duplicates.values())
                
                st.success(f"Duplicate detection completed! Found {len(st.session_state.duplicates)} potential duplicate clusters.")
                
            except Exception as e:
                st.error(f"Error during duplicate detection: {str(e)}")
                st.exception(e)
    
    # Display the duplicate clusters
    if 'duplicates' in st.session_state and st.session_state.duplicates:
        st.markdown(f"**Found {len(st.session_state.duplicates)} potential duplicate clusters.**")
        
        # Initialize merge_choices in session state if it doesn't exist
        if 'merge_choices' not in st.session_state:
            st.session_state.merge_choices = {}
        
        for i, cluster in enumerate(st.session_state.duplicates):
            with st.expander(f"Cluster {i+1} - {len(cluster)} records"):
                st.dataframe(st.session_state.df.iloc[cluster], use_container_width=True)
                
                # Create radio button options for this cluster
                options = ["Keep All (Not Duplicates)"]
                for idx in cluster:
                    options.append(f"Keep Record with Index {idx}")
                
                # Add radio button for user choice
                choice = st.radio(
                    "Choose how to handle this cluster:",
                    options=options,
                    key=f"cluster_{i}",
                    index=0
                )
                
                # Store the user's choice
                st.session_state.merge_choices[i] = choice
        
        # Add merge button after all clusters
        if st.button("Resolve Duplicates and Merge", type="primary"):
            with st.spinner("Processing duplicate resolution..."):
                try:
                    # Create a copy of the main DataFrame
                    df_merged = st.session_state.df.copy()
                    indices_to_drop = []
                    
                    # Process user choices
                    for cluster_idx, choice in st.session_state.merge_choices.items():
                        cluster = st.session_state.duplicates[cluster_idx]
                        
                        if choice.startswith("Keep Record with Index"):
                            # Extract the index of the record to keep
                            keep_index = int(choice.split()[-1])
                            
                            # Add all other records from this cluster to drop list
                            for idx in cluster:
                                if idx != keep_index:
                                    indices_to_drop.append(idx)
                    
                    # Drop the selected records
                    if indices_to_drop:
                        df_merged = df_merged.drop(index=indices_to_drop)
                        st.success(f"✅ Successfully removed {len(indices_to_drop)} duplicate records!")
                    
                    # Update the main DataFrame in session state
                    st.session_state.df = df_merged
                    
                    # Clean up old duplicate information
                    if 'duplicates' in st.session_state:
                        del st.session_state['duplicates']
                    if 'merge_choices' in st.session_state:
                        del st.session_state['merge_choices']
                    
                    # Rerun the app to refresh all displays
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during duplicate resolution: {str(e)}")
                    st.exception(e)
                    
    elif 'duplicates' in st.session_state and not st.session_state.duplicates:
        st.info("No potential duplicates found with the current settings. Try lowering the similarity threshold or selecting different columns.")
else:
    st.info("Upload a CSV file to access duplicate detection features.")

# Section 3: Download Clean Data
st.header("3. Download Clean Data")

if 'df' in st.session_state:
    # Prepare the CSV data for download
    csv_data = convert_df_to_csv(st.session_state.df)
    
    # Create download button
    st.download_button(
        label="📥 Download Cleaned Data as CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv",
        help="Download your cleaned dataset as a CSV file"
    )
    
    # Show summary of current dataset
    st.markdown("**Current Dataset Summary:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", st.session_state.df.shape[0])
    with col2:
        st.metric("Total Columns", st.session_state.df.shape[1])
    with col3:
        missing_count = st.session_state.df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    
    # Add debugging information
    with st.expander("🔍 Debug Information"):
        st.write("**DataFrame Info:**")
        st.write(f"Shape: {st.session_state.df.shape}")
        st.write(f"Memory usage: {st.session_state.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        st.write("**Missing Values by Column:**")
        missing_by_column = st.session_state.df.isnull().sum()
        if missing_by_column.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_by_column.index,
                'Missing Count': missing_by_column.values
            }).query('`Missing Count` > 0')
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values found in any column!")
else:
    st.info("Upload a CSV file to access download functionality.") 