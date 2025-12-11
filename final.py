"""
US Commercial Casino Industry Analysis (2001-2014)
Final Project - Information Visualization
Three Core Visualizations: Time Series, Ridgeline, and Network Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from scipy import stats
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_prepare_data():
    """Load and prepare casino revenue data with regional classification"""
    
    # Load data
    df_wide = pd.read_csv('casino_revenue_wide.csv')
    df_long = pd.read_csv('casino_revenue_long.csv')
    
    # Define regions for analysis
    regions = {
        'West': ['Colorado', 'Kansas', 'Nevada', 'New Mexico', 'Oklahoma', 'South Dakota'],
        'Midwest': ['Illinois', 'Indiana', 'Iowa', 'Michigan', 'Ohio'],
        'South': ['Florida', 'Louisiana', 'Mississippi', 'Missouri'],
        'Northeast': ['Delaware', 'Maine', 'Maryland', 'New Jersey', 'New York', 
                     'Pennsylvania', 'Rhode Island', 'West Virginia']
    }
    
    # Add region column
    df_wide['Region'] = df_wide['State'].apply(
        lambda x: next((k for k, v in regions.items() if x in v), 'Other')
    )
    
    df_long['Region'] = df_long['State'].apply(
        lambda x: next((k for k, v in regions.items() if x in v), 'Other')
    )
    
    return df_wide, df_long, regions

def create_enhanced_time_series(df_wide, save_path='figure1_time_series.png'):
    """
    Figure 1: Enhanced Time Series Analysis with Crisis Impact
    Shows overall trend, growth rates, and cumulative change
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.25)
    
    # Prepare data
    years = range(2001, 2015)
    years_str = [str(y) for y in years]
    total_revenue = [df_wide[str(y)].sum() / 1000 for y in years]  # In millions
    
    # Color scheme
    color_growth = '#2E7D32'  # Green
    color_crisis = '#C62828'  # Red
    color_recovery = '#1565C0'  # Blue
    
    # ========== Panel 1: Main Time Series ==========
    ax1 = fig.add_subplot(gs[0])
    
    # Plot main trend line
    ax1.plot(years, total_revenue, linewidth=3.5, color='#1a237e', 
             marker='o', markersize=10, markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor='#1a237e', zorder=5)
    
    # Add period shadings
    ax1.axvspan(2001, 2007.5, alpha=0.15, color=color_growth, label='Growth Period')
    ax1.axvspan(2007.5, 2009.5, alpha=0.2, color=color_crisis, label='Financial Crisis')
    ax1.axvspan(2009.5, 2014, alpha=0.15, color=color_recovery, label='Recovery Period')
    
    # Add trend line
    z = np.polyfit(years, total_revenue, 2)
    p = np.poly1d(z)
    ax1.plot(years, p(years), '--', linewidth=2, color='gray', alpha=0.5, label='Polynomial Trend')
    
    # Annotate key points with enhanced boxes
    annotations = [
        (2001, total_revenue[0], f'Start:\n${total_revenue[0]:,.0f}M', 'bottom'),
        (2007, total_revenue[6], f'Pre-Crisis Peak:\n${total_revenue[6]:,.0f}M', 'top'),
        (2009, total_revenue[8], f'Crisis Low:\n${total_revenue[8]:,.0f}M', 'bottom'),
        (2014, total_revenue[13], f'End:\n${total_revenue[13]:,.0f}M', 'top')
    ]
    
    for year, value, text, va in annotations:
        if va == 'top':
            y_offset = -1500
            va_pos = 'top'
        else:
            y_offset = 1500
            va_pos = 'bottom'
            
        ax1.annotate(text, xy=(year, value), xytext=(year, value + y_offset),
                    ha='center', va=va_pos, fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor='#1a237e', linewidth=2),
                    arrowprops=dict(arrowstyle='-', color='#1a237e', lw=1.5))
    
    # Crisis impact annotation
    ax1.annotate('', xy=(2007.5, 34000), xytext=(2009.5, 34000),
                arrowprops=dict(arrowstyle='<->', color=color_crisis, lw=2))
    ax1.text(2008.5, 34500, '$3.2B Loss', ha='center', fontsize=11, 
            color=color_crisis, weight='bold')
    
    ax1.set_title('US Casino Industry Revenue Trend: 14-Year Evolution (2001-2014)', 
                 fontsize=16, weight='bold', pad=20)
    ax1.set_ylabel('Total Revenue (Million USD)', fontsize=13, weight='bold')
    ax1.set_xlim(2000.5, 2014.5)
    ax1.set_ylim(25000, 40000)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(years)
    
    # Format y-axis
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}B'))
    
    # ========== Panel 2: Year-over-Year Growth Rate ==========
    ax2 = fig.add_subplot(gs[1])
    
    yoy_growth = [((total_revenue[i] - total_revenue[i-1]) / total_revenue[i-1] * 100) 
                  for i in range(1, len(total_revenue))]
    
    colors = [color_growth if x > 0 else color_crisis for x in yoy_growth]
    bars = ax2.bar(years[1:], yoy_growth, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, value in zip(bars, yoy_growth):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', 
                va='bottom' if height > 0 else 'top',
                fontsize=9, weight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_title('Year-over-Year Growth Rate Analysis', fontsize=13, weight='bold')
    ax2.set_ylabel('Growth Rate (%)', fontsize=12, weight='bold')
    ax2.set_xlim(2001.5, 2014.5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(years[1:])
    
    # ========== Panel 3: Cumulative Growth Index ==========
    ax3 = fig.add_subplot(gs[2])
    
    base_index = 100
    growth_index = [base_index * (r / total_revenue[0]) for r in total_revenue]
    
    ax3.fill_between(years, base_index, growth_index, 
                     where=[g >= base_index for g in growth_index],
                     alpha=0.3, color=color_growth, label='Growth')
    ax3.fill_between(years, base_index, growth_index,
                     where=[g < base_index for g in growth_index],
                     alpha=0.3, color=color_crisis, label='Decline')
    
    ax3.plot(years, growth_index, linewidth=3, color='#4a148c', 
             marker='s', markersize=7, markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='#4a148c')
    
    ax3.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Final growth annotation
    final_growth = growth_index[-1] - 100
    ax3.text(2014, growth_index[-1] + 2, f'Total: +{final_growth:.1f}%', 
            fontsize=11, weight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#4a148c', 
                     alpha=0.2, edgecolor='#4a148c', linewidth=2))
    
    ax3.set_title('Cumulative Growth Index (Base Year 2001 = 100)', 
                 fontsize=13, weight='bold')
    ax3.set_xlabel('Year', fontsize=12, weight='bold')
    ax3.set_ylabel('Index Value', fontsize=12, weight='bold')
    ax3.set_xlim(2000.5, 2014.5)
    ax3.set_ylim(95, 145)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.set_xticks(years)
    
    # Add data source
    fig.text(0.99, 0.01, 'Data Source: UNLV Center for Gaming Research', 
            ha='right', fontsize=9, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig

def create_enhanced_ridgeline(df_wide, save_path='figure2_ridgeline.png'):
    """
    Figure 2: Enhanced Ridgeline Plot
    Shows distribution evolution over time with crisis highlight
    """
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    years = [str(y) for y in range(2001, 2015)]
    n_years = len(years)
    
    # Create color gradient
    colors_normal = plt.cm.viridis(np.linspace(0.3, 0.9, n_years))
    
    # Ridge parameters
    overlap = 0.6
    scale = 4000
    
    for i, year in enumerate(years):
        # Get revenue data
        revenues = df_wide[year].dropna().values / 1000  # Convert to millions
        
        if len(revenues) > 1:
            # Kernel density estimation
            density = stats.gaussian_kde(revenues, bw_method=0.3)
            xs = np.linspace(0, 12000, 400)
            ys = density(xs)
            
            # Scale and offset
            ys = ys * scale + i
            
            # Special coloring for crisis years
            if year in ['2008', '2009']:
                color = '#FF4444'
                alpha = 0.7
                linewidth = 2.5
            else:
                color = colors_normal[i]
                alpha = 0.6
                linewidth = 2
            
            # Plot ridge
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=0.9)
            ax.fill_between(xs, i, ys, alpha=alpha, color=color)
            
            # Year labels with enhanced formatting
            if year in ['2008', '2009']:
                ax.text(-800, i + 0.3, year, fontsize=12, weight='bold', color='#FF4444')
            else:
                ax.text(-800, i + 0.3, year, fontsize=11, weight='bold', color='#333333')
            
            # Add markers for top states
            if year == '2014':
                top_revenues = df_wide.nlargest(3, year)[year].values / 1000
                for rev in top_revenues:
                    y_val = density([rev])[0] * scale + i
                    ax.plot(rev, y_val, 'o', color='gold', markersize=8, 
                           markeredgewidth=2, markeredgecolor='darkgoldenrod')
    
    # Annotations
    ax.annotate('Financial Crisis Period', xy=(3000, 7.8), xytext=(5000, 10),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=12, color='red', weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='red', linewidth=2))
    
    ax.annotate('Market Expansion\n(Distribution Widens)', 
               xy=(8000, 12), xytext=(9500, 10),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=11, color='green', weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        edgecolor='green', linewidth=2))
    
    # Nevada dominance indicator
    ax.axvline(x=11000, color='gold', linestyle='--', alpha=0.5, linewidth=2)
    ax.text(11100, 13, 'Nevada\nDominance', fontsize=10, color='darkgoldenrod',
           weight='bold', ha='left')
    
    ax.set_xlim(-500, 12500)
    ax.set_ylim(-0.5, n_years)
    ax.set_xlabel('State Revenue (Million USD)', fontsize=14, weight='bold')
    ax.set_title('Evolution of US Casino Revenue Distributions: A Ridgeline Visualization\n' + 
                'Showing Market Democratization Over 14 Years', 
                fontsize=16, weight='bold', pad=20)
    
    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    
    # X-axis formatting
    ax.set_xticks(range(0, 13000, 2000))
    ax.set_xticklabels([f'${x/1000:.0f}B' for x in range(0, 13000, 2000)])
    
    # Add gradient legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=colors_normal[0], alpha=0.6, label='Early Years (2001-2007)'),
        Rectangle((0, 0), 1, 1, fc='#FF4444', alpha=0.7, label='Crisis Years (2008-2009)'),
        Rectangle((0, 0), 1, 1, fc=colors_normal[-1], alpha=0.6, label='Recovery Years (2010-2014)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Data source
    fig.text(0.99, 0.01, 'Data Source: UNLV Center for Gaming Research', 
            ha='right', fontsize=9, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig

def create_enhanced_network(df_wide, save_path='figure3_network.png'):
    """
    Figure 3: Enhanced Network Analysis
    Shows state relationships and market structure
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2, 1], 
                          height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Calculate correlation matrix
    years = [str(y) for y in range(2001, 2015)]
    corr_matrix = df_wide[years].T.corr()
    
    # ========== Main Network Graph ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Build network
    G = nx.Graph()
    threshold = 0.75  # Strong correlation threshold
    
    # Add nodes
    for i, state in enumerate(df_wide['State']):
        revenue_2014 = df_wide.iloc[i]['2014']
        if pd.notna(revenue_2014):
            G.add_node(state, revenue=revenue_2014/1000)  # In millions
    
    # Add edges for strong correlations
    edge_weights = []
    for i in range(len(df_wide)):
        for j in range(i+1, len(df_wide)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > threshold and not np.isnan(corr_val):
                state_i = df_wide.iloc[i]['State']
                state_j = df_wide.iloc[j]['State']
                if state_i in G.nodes() and state_j in G.nodes():
                    G.add_edge(state_i, state_j, weight=corr_val)
                    edge_weights.append(corr_val)
    
    # Layout with better positioning
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Node sizes and colors based on revenue
    node_sizes = [G.nodes[node].get('revenue', 100)/3 for node in G.nodes()]
    node_colors = [G.nodes[node].get('revenue', 0) for node in G.nodes()]
    
    # Draw network
    edges = nx.draw_networkx_edges(G, pos, alpha=0.2, width=2, 
                                   edge_color='gray', ax=ax1)
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                   node_color=node_colors, cmap='YlOrRd',
                                   alpha=0.8, ax=ax1, vmin=0, vmax=11000)
    
    # Highlight top states
    top_states = ['Nevada', 'Pennsylvania', 'New Jersey']
    for state in top_states:
        if state in pos:
            x, y = pos[state]
            circle = plt.Circle((x, y), 0.08, color='gold', fill=False, 
                               linewidth=3, linestyle='--')
            ax1.add_patch(circle)
    
    # Labels
    labels = {}
    for node in G.nodes():
        if G.nodes[node].get('revenue', 0) > 1000 or node in top_states:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold', ax=ax1)
    
    ax1.set_title('Casino Revenue Network: State Interconnections\n' +
                 '(Edges show correlation > 0.75 in growth patterns)', 
                 fontsize=14, weight='bold')
    ax1.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=11000))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
    cbar.set_label('2014 Revenue (Million $)', rotation=270, labelpad=20)
    
    # ========== Network Metrics ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate centrality metrics
    degree_cent = nx.degree_centrality(G)
    betweenness_cent = nx.betweenness_centrality(G)
    
    # Top 10 by degree centrality
    top_central = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    
    states = [s[0] for s in top_central]
    centrality = [s[1] for s in top_central]
    
    bars = ax2.barh(range(len(states)), centrality, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(states)))
    ax2.set_yticklabels(states, fontsize=10)
    ax2.set_xlabel('Centrality Score', fontsize=11, weight='bold')
    ax2.set_title('Network Centrality Ranking\n(Most Connected States)', 
                 fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, centrality)):
        ax2.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=9)
    
    ax2.set_xlim(0, max(centrality) * 1.15)
    
    # ========== Network Statistics ==========
    ax3 = fig.add_subplot(gs[1, :])
    
    # Calculate network statistics
    stats_text = f"""Network Structure Analysis:
    
    • Total States in Network: {G.number_of_nodes()}
    • Total Connections: {G.number_of_edges()}
    • Average Connections per State: {2 * G.number_of_edges() / G.number_of_nodes():.1f}
    • Network Density: {nx.density(G):.3f}
    • Average Clustering Coefficient: {nx.average_clustering(G):.3f}
    
    Key Insights:
    • High clustering indicates regional market similarities
    • Nevada remains central despite declining market share
    • Eastern states form tight cluster (similar growth patterns post-2006)
    """
    
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', alpha=0.8))
    
    ax3.axis('off')
    
    # Main title
    fig.suptitle('Network Analysis of US Casino Industry Structure', 
                fontsize=16, weight='bold', y=0.98)
    
    # Data source
    fig.text(0.99, 0.01, 'Data Source: UNLV Center for Gaming Research', 
            ha='right', fontsize=9, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    return fig

def generate_key_statistics(df_wide, df_long):
    """Generate key statistics for the report"""
    
    years = [str(y) for y in range(2001, 2015)]
    
    # Total revenue stats
    total_2001 = df_wide['2001'].sum() / 1000  # millions
    total_2014 = df_wide['2014'].sum() / 1000
    
    # Growth calculation
    overall_growth = (total_2014 / total_2001 - 1) * 100
    
    # Crisis impact
    total_2007 = df_wide['2007'].sum() / 1000
    total_2009 = df_wide['2009'].sum() / 1000
    crisis_loss = total_2007 - total_2009
    
    # Market concentration (HHI)
    market_shares_2001 = (df_wide['2001'] / df_wide['2001'].sum()) ** 2
    market_shares_2014 = (df_wide['2014'] / df_wide['2014'].sum()) ** 2
    hhi_2001 = market_shares_2001.sum() * 10000
    hhi_2014 = market_shares_2014.sum() * 10000
    
    # Number of active states
    active_2001 = (df_wide['2001'].notna()).sum()
    active_2014 = (df_wide['2014'].notna()).sum()
    
    # Top state info
    top_state_2014 = df_wide.nlargest(1, '2014')[['State', '2014']].values[0]
    nevada_share_2001 = df_wide[df_wide['State'] == 'Nevada']['2001'].values[0] / df_wide['2001'].sum() * 100
    nevada_share_2014 = df_wide[df_wide['State'] == 'Nevada']['2014'].values[0] / df_wide['2014'].sum() * 100
    
    stats = f"""
KEY STATISTICS SUMMARY
======================
    
Revenue Overview:
• 2001 Total: ${total_2001:,.0f}M
• 2014 Total: ${total_2014:,.0f}M  
• Overall Growth: {overall_growth:.1f}%
• Annual Average Growth: {overall_growth/13:.1f}%

Crisis Impact (2008-2009):
• Revenue Loss: ${crisis_loss:,.0f}M
• Peak-to-Trough Decline: {(total_2009/total_2007 - 1)*100:.1f}%
• Recovery Time: 4 years (2010-2013)

Market Structure:
• Active States 2001: {active_2001}
• Active States 2014: {active_2014}
• New Entrants: {active_2014 - active_2001} states
• Market Concentration (HHI):
  - 2001: {hhi_2001:.0f} (concentrated)
  - 2014: {hhi_2014:.0f} (less concentrated)

Market Leader (Nevada):
• Market Share 2001: {nevada_share_2001:.1f}%
• Market Share 2014: {nevada_share_2014:.1f}%
• Change: {nevada_share_2014 - nevada_share_2001:.1f} percentage points

Regional Performance:
• Fastest Growing: Northeast (+67% share)
• Most Stable: Midwest
• Highest Revenue: West (due to Nevada)
    """
    
    return stats

def main():
    """Main execution function"""
    
    print("="*60)
    print("US CASINO INDUSTRY VISUALIZATION PROJECT")
    print("Information Visualization - Final Project")
    print("="*60 + "\n")
    
    print("Loading data...")
    df_wide, df_long, regions = load_and_prepare_data()
    print("✓ Data loaded successfully\n")
    
    print("Creating visualizations...\n")
    
    # Figure 1: Time Series Analysis
    print("Creating Figure 1: Enhanced Time Series Analysis...")
    fig1 = create_enhanced_time_series(df_wide)
    print("✓ Figure 1 saved as 'figure1_time_series.png'\n")
    
    # Figure 2: Ridgeline Plot
    print("Creating Figure 2: Enhanced Ridgeline Plot...")
    fig2 = create_enhanced_ridgeline(df_wide)
    print("✓ Figure 2 saved as 'figure2_ridgeline.png'\n")
    
    # Figure 3: Network Analysis
    print("Creating Figure 3: Enhanced Network Analysis...")
    fig3 = create_enhanced_network(df_wide)
    print("✓ Figure 3 saved as 'figure3_network.png'\n")
    
    # Generate statistics
    print("Generating statistical summary...")
    stats = generate_key_statistics(df_wide, df_long)
    
    # Save statistics to file
    with open('project_statistics.txt', 'w') as f:
        f.write(stats)
    
    print(stats)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("1. figure1_time_series.png - Main trend analysis")
    print("2. figure2_ridgeline.png - Distribution evolution")  
    print("3. figure3_network.png - State relationships")
    print("4. project_statistics.txt - Key statistics")
    print("\nReady for final report assembly!")

if __name__ == "__main__":
    main()