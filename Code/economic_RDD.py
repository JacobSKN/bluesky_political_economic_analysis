import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import os
from datetime import datetime, timedelta
import json

def run_rdd_analysis(economic_posts, analysis, 
                    event_date='2025-02-23',
                    bandwidth_days=7,
                    polynomial_order=1,
                    rolling_window=7,
                    output_dir='economic_topic_rdd_plots'):
    """
    Run Regression Discontinuity Design analysis on economic topics data.
    
    Args:
        economic_posts (pd.DataFrame): DataFrame containing posts with economic content
        analysis (dict): Analysis results containing categories
        event_date (str): Date of the event to analyze discontinuity
        bandwidth_days (int): Number of days before and after event date to include
        polynomial_order (int): Order of polynomial for RDD regression
        rolling_window (int): Window size for rolling average of percentages
        output_dir (str): Directory to save output files
    
    Returns:
        dict: Results of RDD analysis for each economic topic
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    economic_posts = economic_posts.copy()
    economic_posts['timestamp'] = pd.to_datetime(economic_posts['timestamp'])
    economic_posts['date'] = pd.to_datetime(economic_posts['timestamp']).dt.date
    
    # Convert event date to datetime
    event_date = pd.to_datetime(event_date).date()
    
    # Define time window for RDD
    start_date = event_date - timedelta(days=bandwidth_days)
    end_date = event_date + timedelta(days=bandwidth_days)
    
    # Extract economic topic categories
    topics = list(analysis['categories'].keys())
    
    # Prepare data structure for results
    rdd_results = {}
    
    # Calculate daily topic volumes
    daily_counts, total_daily_counts, relative_volumes = analyze_economic_topic_volumes(
        economic_posts, analysis,
        event_date=event_date.strftime('%Y-%m-%d'),
        baseline_start=(event_date - timedelta(days=bandwidth_days)).strftime('%Y-%m-%d'),
        baseline_end=(event_date - timedelta(days=1)).strftime('%Y-%m-%d'),
        display_start=(event_date - timedelta(days=30)).strftime('%Y-%m-%d'),
        display_end=(event_date + timedelta(days=30)).strftime('%Y-%m-%d')
    )
    
    # Create RDD dataset for each topic
    for topic in topics:
        # Get topic relative volumes
        topic_volumes = relative_volumes[topic]
        
        # Calculate rolling average
        if rolling_window > 1:
            topic_volumes_rolling = topic_volumes.rolling(window=rolling_window, center=True, min_periods=1).mean()
        else:
            topic_volumes_rolling = topic_volumes
            
        # Filter to RDD time window
        rdd_data = topic_volumes_rolling[(topic_volumes_rolling.index >= start_date) & 
                                       (topic_volumes_rolling.index <= end_date)]
        
        if rdd_data.empty:
            print(f"Warning: No data for topic {topic} in the specified time window")
            continue
            
        # Create dataframe for RDD analysis
        rdd_df = pd.DataFrame({
            'date': rdd_data.index,
            'volume': rdd_data.values,
            'days_from_event': [(d - event_date).days for d in rdd_data.index],
            'post_event': [(d >= event_date) * 1.0 for d in rdd_data.index]  # Convert boolean to numeric
        })
        
        # Run RDD analysis
        rdd_results[topic] = run_single_topic_rdd(rdd_df, topic, event_date, 
                                               polynomial_order, output_dir)
        
    # Save aggregate results
    save_aggregate_rdd_results(rdd_results, event_date, bandwidth_days, 
                             polynomial_order, rolling_window, output_dir)
    
    return rdd_results

def run_single_topic_rdd(rdd_df, topic, event_date, polynomial_order, output_dir):
    """
    Run RDD analysis for a single economic topic.
    
    Args:
        rdd_df (pd.DataFrame): DataFrame with RDD data for this topic
        topic (str): Name of the economic topic
        event_date (datetime.date): Date of the event
        polynomial_order (int): Order of polynomial for RDD regression
        output_dir (str): Directory to save output files
    
    Returns:
        dict: Results of RDD analysis for this topic
    """
    # Pretty topic name (replace underscores with spaces and capitalize)
    topic_name = ' '.join(word.capitalize() for word in topic.split('_'))
    
    # Create polynomial terms based on order
    for i in range(1, polynomial_order + 1):
        rdd_df[f'days_from_event_{i}'] = rdd_df['days_from_event'] ** i
    
    # Add interaction terms
    for i in range(1, polynomial_order + 1):
        rdd_df[f'post_event_days_{i}'] = rdd_df['post_event'] * rdd_df[f'days_from_event_{i}']
    
    try:
        # Prepare X matrix
        X_columns = ['post_event']
        for i in range(1, polynomial_order + 1):
            X_columns.append(f'days_from_event_{i}')
            X_columns.append(f'post_event_days_{i}')
        
        X = rdd_df[X_columns]
        X = sm.add_constant(X)  # Add intercept
        y = rdd_df['volume']
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Calculate predicted values
        rdd_df['predicted'] = model.predict(X)
        
        # Calculate discontinuity estimate (coefficient on post_event)
        discontinuity = model.params['post_event']
        p_value = model.pvalues['post_event']
        
        # Calculate means before and after event
        pre_mean = rdd_df[rdd_df['post_event'] == 0]['volume'].mean()
        post_mean = rdd_df[rdd_df['post_event'] == 1]['volume'].mean()
        percent_change = ((post_mean - pre_mean) / pre_mean) * 100 if pre_mean != 0 else np.nan
        
        # Plot RDD
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot of actual data
        ax.scatter(rdd_df['days_from_event'], rdd_df['volume'], 
                 alpha=0.6, label='Observed data')
        
        # Split data for prediction lines
        pre_event = rdd_df[rdd_df['post_event'] == 0].sort_values('days_from_event')
        post_event = rdd_df[rdd_df['post_event'] == 1].sort_values('days_from_event')
        
        # Plot prediction lines
        if not pre_event.empty:
            ax.plot(pre_event['days_from_event'], pre_event['predicted'], 
                   'b-', linewidth=2, label='Pre-event fit')
        if not post_event.empty:
            ax.plot(post_event['days_from_event'], post_event['predicted'], 
                   'r-', linewidth=2, label='Post-event fit')
        
        # Add vertical line at event
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.7, label='Event date')
        
        # Add significance indicator in title
        sig_marker = ""
        if p_value < 0.01:
            sig_marker = "***"
        elif p_value < 0.05:
            sig_marker = "**"
        elif p_value < 0.1:
            sig_marker = "*"
        
        # Add text annotations
        significance_text = f"Significant at: {'p<0.01' if p_value < 0.01 else 'p<0.05' if p_value < 0.05 else 'p<0.1' if p_value < 0.1 else 'Not significant'}"
        ax.text(0.05, 0.95, f"Discontinuity: {discontinuity:.2f}% {sig_marker}", 
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.05, 0.89, f"Pre-event mean: {pre_mean:.2f}%", 
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.05, 0.83, f"Post-event mean: {post_mean:.2f}%", 
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.05, 0.77, f"Change: {percent_change:.2f}%", 
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        ax.text(0.05, 0.71, significance_text, 
               transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Set title and labels
        ax.set_title(f"{topic_name} - RDD Analysis (Event Date: {event_date}){' ' + sig_marker if sig_marker else ''}")
        ax.set_xlabel("Days from Event")
        ax.set_ylabel("Percentage of Total Economic Volume (Rolling Avg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{topic}_rdd_analysis.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create result dictionary
        result = {
            'topic': topic,
            'topic_name': topic_name,
            'discontinuity': discontinuity,
            'p_value': p_value,
            'pre_mean': pre_mean,
            'post_mean': post_mean,
            'percent_change': percent_change,
            'significant': p_value < 0.05,
            'significance_level': '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.1 else '',
            'r_squared': model.rsquared,
            'model_summary': model.summary().as_text()  # Convert summary to text to make it serializable
        }
        
        return result
        
    except Exception as e:
        print(f"Error in RDD analysis for topic {topic}: {str(e)}")
        return {
            'topic': topic,
            'topic_name': topic_name,
            'error': str(e)
        }

def save_aggregate_rdd_results(rdd_results, event_date, bandwidth_days, 
                             polynomial_order, rolling_window, output_dir):
    """
    Save aggregate results from RDD analysis.
    
    Args:
        rdd_results (dict): Results of RDD analysis for each topic
        event_date (datetime.date): Date of the event
        bandwidth_days (int): Number of days in bandwidth
        polynomial_order (int): Order of polynomial used
        rolling_window (int): Window size for rolling average
        output_dir (str): Directory to save output
    """
    # Filter out topics with errors
    valid_results = {topic: results for topic, results in rdd_results.items() 
                    if 'error' not in results}
    
    if not valid_results:
        print("No valid RDD results to summarize")
        return
    
    # Create summary dataframe
    summary_data = []
    for topic, results in valid_results.items():
        summary_data.append({
            'Topic': results['topic_name'],
            'Discontinuity': results['discontinuity'],
            'P-value': results['p_value'],
            'Pre-Mean': results['pre_mean'],
            'Post-Mean': results['post_mean'],
            'Percent Change': results['percent_change'],
            'Significant': results['significant'],
            'Significance': results['significance_level']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Sort by absolute discontinuity (descending)
    summary_df = summary_df.sort_values('Discontinuity', key=abs, ascending=False)
    
    # Save to CSV
    summary_df.to_csv(f"{output_dir}/rdd_summary.csv", index=False)
    
    # Create summary bar chart of discontinuities
    plt.figure(figsize=(12, 8))
    
    # Create color-coded bars based on significance
    colors = []
    for _, row in summary_df.iterrows():
        if row['Significant']:
            colors.append('green')
        else:
            colors.append('gray')
    
    # Create bar chart
    bars = plt.barh(summary_df['Topic'], summary_df['Discontinuity'], color=colors)
    
    # Add labels at the end of each bar
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + (0.5 if bar.get_width() >= 0 else -0.5), 
            bar.get_y() + bar.get_height()/2,
            f"{summary_df.iloc[i]['Discontinuity']:.2f}% {summary_df.iloc[i]['Significance']}",
            va='center'
        )
    
    # Add a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # Add title and labels
    plt.title(f"RDD Discontinuity Estimates (Event: {event_date}, Bandwidth: {bandwidth_days} days)")
    plt.xlabel("Estimated Discontinuity (percentage points)")
    plt.ylabel("Economic Topic")
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend
    plt.legend([
        plt.Rectangle((0,0),1,1, color='green'),
        plt.Rectangle((0,0),1,1, color='gray')
    ], ['Significant (p<0.05)', 'Not Significant'], loc='lower right')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rdd_summary_chart.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create parameter summary
    with open(f"{output_dir}/rdd_parameters.txt", 'w') as f:
        f.write(f"RDD Analysis Parameters\n")
        f.write(f"=======================\n\n")
        f.write(f"Event Date: {event_date}\n")
        f.write(f"Bandwidth: {bandwidth_days} days before and after event\n")
        f.write(f"Polynomial Order: {polynomial_order}\n")
        f.write(f"Rolling Window: {rolling_window} days\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def analyze_economic_topic_volumes(economic_posts, analysis, 
                                   event_date='2025-02-23', 
                                   baseline_start='2025-02-05', 
                                   baseline_end='2025-02-12',
                                   display_start='2025-01-15', 
                                   display_end='2025-02-28'):
    """
    Analyze post volumes for economic topics as a percentage of total economic volume,
    normalized to ensure percentages sum to 100%.
    """
    # Ensure we have datetime objects
    economic_posts = economic_posts.copy()
    economic_posts['timestamp'] = pd.to_datetime(economic_posts['timestamp'])
    economic_posts['date'] = economic_posts['timestamp'].dt.date
    
    # Create a complete date range for the display period
    date_range = pd.date_range(start=display_start, end=display_end)
    
    # Extract economic topic categories
    topics = list(analysis['categories'].keys())
    
    # Convert dates to datetime
    event_date = pd.to_datetime(event_date).date()
    baseline_start = pd.to_datetime(baseline_start).date()
    baseline_end = pd.to_datetime(baseline_end).date()
    
    # Function to check if a post contains terms from a specific category
    def post_contains_category_terms(row, category):
        terms = row['economic_terms_found']
        # Handle string representation of lists
        if isinstance(terms, str):
            try:
                terms = eval(terms)
            except:
                terms = []
                
        if not terms:
            return False
            
        # Get terms in this category
        category_terms = [item['term'] for item in analysis['categories'][category]]
        
        # Check if any term from this post is in the category
        return any(term in category_terms for term in terms)
    
    # Dictionary to store daily counts
    daily_counts = {}
    total_daily_counts = pd.Series(0, index=date_range.date)
    
    # Get counts for each economic topic
    for topic in topics:
        # Filter posts containing terms from this category
        topic_mask = economic_posts.apply(lambda row: post_contains_category_terms(row, topic), axis=1)
        topic_posts = economic_posts[topic_mask]
        
        # Calculate daily counts
        daily_count = topic_posts.groupby('date').size()
        daily_count = daily_count.reindex(date_range.date, fill_value=0)
        daily_counts[topic] = daily_count
    
    # For total counts, use the total number of economic posts per day
    total_economic_daily = economic_posts.groupby('date').size()
    total_economic_daily = total_economic_daily.reindex(date_range.date, fill_value=0)
    
    # Calculate the sum of all topic counts for each day
    topic_counts_sum = pd.Series(0, index=date_range.date)
    for topic in topics:
        topic_counts_sum += daily_counts[topic]
    
    # Calculate normalized percentages (ensuring sum = 100%)
    relative_volumes = {}
    for topic in topics:
        # Avoid division by zero
        safe_sum = topic_counts_sum.copy()
        safe_sum[safe_sum == 0] = 1  # Replace zeros with ones to avoid division errors
        
        # Calculate normalized percentage
        relative_volumes[topic] = (daily_counts[topic] / safe_sum) * 100
    
    return daily_counts, total_economic_daily, relative_volumes

# Example usage
def run_rdd_with_parameters(economic_posts, analysis, event_date, bandwidth_days, 
                           polynomial_order=1, rolling_window=7, output_dir='economic_topic_rdd_plots'):
    """
    Wrapper function to run RDD analysis with specified parameters
    
    Args:
        economic_posts (pd.DataFrame): DataFrame containing posts with economic content
        analysis (dict): Analysis results containing categories
        event_date (str): Date of the event to analyze discontinuity (YYYY-MM-DD)
        bandwidth_days (int): Number of days before and after event date to include
        polynomial_order (int): Order of polynomial for RDD regression (1 = linear, 2 = quadratic)
        rolling_window (int): Window size for rolling average of percentages
        output_dir (str): Directory to save output files
        
    Returns:
        dict: Results of RDD analysis
    """
    # Create versioned output directory based on parameters
    versioned_dir = f"{output_dir}/event_{event_date}_bw{bandwidth_days}_poly{polynomial_order}_roll{rolling_window}"
    
    # Run the analysis
    results = run_rdd_analysis(
        economic_posts=economic_posts,
        analysis=analysis,
        event_date=event_date,
        bandwidth_days=bandwidth_days,
        polynomial_order=polynomial_order,
        rolling_window=rolling_window,
        output_dir=versioned_dir
    )
    
    print(f"RDD analysis completed. Results saved to: {versioned_dir}")
    
    # Print summary of significant findings
    significant_results = {k: v for k, v in results.items() 
                          if 'error' not in v and v.get('significant', False)}
    
    if significant_results:
        print("\nSignificant discontinuities found:")
        for topic, result in significant_results.items():
            print(f"- {result['topic_name']}: {result['discontinuity']:.2f}% " +
                 f"({result['significance_level']}, p={result['p_value']:.4f})")
    else:
        print("\nNo significant discontinuities found.")
        
    return results

# Main function to execute
if __name__ == "__main__":
    # Load your data
    economic_posts = pd.read_csv("H:/bluesky_data/political_economic_analysis/20250311_131554/political_economic_posts.csv")
    with open("H:/bluesky_data/political_economic_analysis/20250311_131554/political_economic_analysis.json", 'r') as f:
        analysis = json.load(f)
    
    # Run RDD with various parameters
    # Example 1: Election day with 7-day bandwidth
    run_rdd_with_parameters(
        economic_posts=economic_posts,
        analysis=analysis,
        event_date="2025-02-23",  # Election day
        bandwidth_days=7,
        polynomial_order=1,
        rolling_window=7,
        output_dir="C:/Users/jsc/Documents/Seafile/Bluesky_Results/economic_analysis"
    )
    
    # Example 2: Same event with 14-day bandwidth
    run_rdd_with_parameters(
        economic_posts=economic_posts,
        analysis=analysis,
        event_date="2025-02-23",
        bandwidth_days=14,
        polynomial_order=1,
        rolling_window=7,
        output_dir="C:/Users/jsc/Documents/Seafile/Bluesky_Results/economic_analysis"
    )

    run_rdd_with_parameters(
        economic_posts=economic_posts,
        analysis=analysis,
        event_date="2025-02-23",  # Election day
        bandwidth_days=7,
        polynomial_order=1,
        rolling_window=14,
        output_dir="C:/Users/jsc/Documents/Seafile/Bluesky_Results/economic_analysis"
    )
    
    # Example 2: Same event with 14-day bandwidth
    run_rdd_with_parameters(
        economic_posts=economic_posts,
        analysis=analysis,
        event_date="2025-02-23",
        bandwidth_days=14,
        polynomial_order=1,
        rolling_window=14,
        output_dir="C:/Users/jsc/Documents/Seafile/Bluesky_Results/economic_analysis"
    )

    run_rdd_with_parameters(
        economic_posts=economic_posts,
        analysis=analysis,
        event_date="2025-02-23",  # Election day
        bandwidth_days=7,
        polynomial_order=1,
        rolling_window=3,
        output_dir="C:/Users/jsc/Documents/Seafile/Bluesky_Results/economic_analysis"
    )
    
    # Example 2: Same event with 14-day bandwidth
    run_rdd_with_parameters(
        economic_posts=economic_posts,
        analysis=analysis,
        event_date="2025-02-23",
        bandwidth_days=14,
        polynomial_order=1,
        rolling_window=3,
        output_dir="C:/Users/jsc/Documents/Seafile/Bluesky_Results/economic_analysis"
    )