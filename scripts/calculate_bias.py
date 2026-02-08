"""
Bias Analysis Script for Triage Counterfactual Experiments

Generates a LaTeX table and comprehensive statistics summary for demographic 
bias in LLM triage predictions.

Usage:
    python calculate_bias.py                    # Default: LaTeX table + summary
    python calculate_bias.py --output-tex out.tex  # Save LaTeX to file
"""

import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, mannwhitneyu, kruskal
import argparse
from pathlib import Path

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'Triage-Counterfactual'

# Models to analyze (name, filename)
MODELS = [
    ('Llama 3.1-70B', 'Triage-Counterfactual_Vanilla_meta_llama_v3_1_70b_instruct_natural_0_2300_2025-02-27_18:39:27.csv'),
    ('Gemini 2.0', 'Triage-Counterfactual_Vanilla_gemini-2.0-flash-001_json_natural_0_2300_2025-02-28_00:39:59.csv'),
    ('4o-mini Van', 'Triage-Counterfactual_ZeroShot_gpt-4o-mini_json0_10000_20241123_123030.csv'),
    ('4o-mini CoT', 'Triage-Counterfactual_CoT_gpt-4o-mini_json0_10000_20241123_142508.csv'),
    ('4o Van', 'Triage-Counterfactual_ZeroShot_openai-gpt-4o-chat_json0_3000_20241124_095918.csv'),
    ('4o CoT', 'Triage-Counterfactual_CoT_gpt-4o_json0_3000.csv'),
    ('Haiku Van', 'Triage-Counterfactual_ZeroShot_claude-3-haiku-20240307_json0_3000_20241127_034105.csv'),
    ('Haiku CoT', 'Triage-Counterfactual_CoT_claude-3-haiku-20240307_json0_3000_20241127_032349.csv'),
    ('Sonnet Van', 'Triage-Counterfactual_ZeroShot_claude-3-sonnet-20240229_json0_3000_20241127_072127.csv'),
    ('Sonnet CoT', 'Triage-Counterfactual_CoT_claude-3-sonnet-20240229_json0_3000_20241127_065552.csv'),
    ('o3-mini', 'Triage-Counterfactual_Vanilla_openai-o3-mini-chat_json_natural_0_2300_2025-02-27_19:15:01.csv'),
]

RACES = ['American Indian', 'Asian', 'Black', 'Hispanic', 
         'Native Hawaiian and Other Pacific Islander', 'White']

RACE_SHORT = {
    'American Indian': 'American Indian',
    'Asian': 'Asian',
    'Black': 'Black',
    'Hispanic': 'Hispanic',
    'Native Hawaiian and Other Pacific Islander': 'Native Hawaiian',
    'White': 'White'
}


def load_data(filepath: str) -> tuple[pd.DataFrame, dict]:
    """Load and preprocess results CSV. Returns (df, info_dict)."""
    df = pd.read_csv(filepath)
    info = {'original_rows': len(df)}
    
    # Normalize column names
    if 'Race' in df.columns:
        df = df.rename(columns={'Race': 'race', 'Sex': 'gender'})
    
    # Track dropped samples
    before_dropna = len(df)
    df = df.dropna(subset=['Estimated_Acuity'])
    info['dropped_na'] = before_dropna - len(df)
    info['final_rows'] = len(df)
    
    df['gender'] = df['gender'].replace({'Men': 'M', 'Women': 'F'})
    df['Group'] = df['race'] + "_" + df['gender']
    
    # Check for stay_id for proper pairing
    info['has_stay_id'] = 'stay_id' in df.columns
    
    return df, info


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute all statistics for a single model using proper stay_id pairing."""
    stats = {'n': len(df)}
    
    # Means by gender
    for g in ['M', 'F']:
        gdf = df[df['gender'] == g]
        stats[f'mean_{g}'] = gdf['Estimated_Acuity'].mean()
        for race in RACES:
            rdf = gdf[gdf['race'] == race]
            stats[f'mean_{g}_{race}'] = rdf['Estimated_Acuity'].mean()
    
    # =========================================================================
    # STATISTICAL TESTS - Using proper stay_id pairing
    # =========================================================================
    
    # 1) SEX TEST: Wilcoxon signed-rank (paired by stay_id)
    # Pivot to get Men vs Women for the same patient
    try:
        sex_pivot = df.pivot_table(index='stay_id', columns='gender', 
                                    values='Estimated_Acuity', aggfunc='mean')
        sex_pivot = sex_pivot.dropna()  # Drop patients missing either gender
        _, stats['p_sex'] = wilcoxon(sex_pivot['M'], sex_pivot['F'])
        stats['n_sex_pairs'] = len(sex_pivot)
    except Exception as e:
        stats['p_sex'] = 1.0
        stats['n_sex_pairs'] = 0
    
    # 2) RACE TEST: Friedman (paired by stay_id across 6 race conditions)
    try:
        race_pivot = df.pivot_table(index='stay_id', columns='race',
                                     values='Estimated_Acuity', aggfunc='mean')
        race_pivot = race_pivot.dropna()  # Need all 6 races per patient
        race_arrays = [race_pivot[r].values for r in RACES]
        _, stats['p_race'] = friedmanchisquare(*race_arrays)
        stats['n_race_pairs'] = len(race_pivot)
    except Exception as e:
        stats['p_race'] = 1.0
        stats['n_race_pairs'] = 0
    
    # 3) COMBINED TEST: Friedman (paired by stay_id across 12 groups)
    try:
        combined_pivot = df.pivot_table(index='stay_id', columns='Group',
                                         values='Estimated_Acuity', aggfunc='mean')
        combined_pivot = combined_pivot.dropna()  # Need all 12 groups per patient
        group_arrays = [combined_pivot[g].values for g in combined_pivot.columns]
        _, stats['p_combined'] = friedmanchisquare(*group_arrays)
        stats['n_combined_pairs'] = len(combined_pivot)
    except Exception as e:
        stats['p_combined'] = 1.0
        stats['n_combined_pairs'] = 0
    
    # =========================================================================
    # BONFERRONI CORRECTION (3 tests)
    # =========================================================================
    raw_pvals = [stats['p_sex'], stats['p_race'], stats['p_combined']]
    adjusted = [min(p * 3, 1.0) for p in raw_pvals]  # Bonferroni: multiply by # tests
    stats['p_sex_adj'] = adjusted[0]
    stats['p_race_adj'] = adjusted[1]
    stats['p_combined_adj'] = adjusted[2]
    
    # Prediction variance per patient (instability metric)
    try:
        variance_per_patient = df.groupby('stay_id')['Estimated_Acuity'].var()
        stats['mean_variance'] = variance_per_patient.mean()
        stats['max_variance'] = variance_per_patient.max()
    except:
        stats['mean_variance'] = 0
        stats['max_variance'] = 0
    
    return stats


def format_val(val, is_min, is_max):
    """Format value with bold (min) or underline (max)."""
    s = f"{val:.3f}"
    if is_min:
        return f"\\textbf{{{s}}}"
    if is_max:
        return f"\\underline{{{s}}}"
    return s


def format_p(p):
    """Format p-value for LaTeX."""
    if p < 0.01:
        return "$<0.01^{**}$"
    if p < 0.05:
        return f"{p:.2f}$^{{*}}$"
    return f"{p:.2f}" if p < 0.995 else "1.00"


def generate_latex(all_stats: dict) -> str:
    """Generate LaTeX table."""
    models = list(all_stats.keys())
    n = len(models)
    
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Mean Estimated Acuity by Demographic Group}",
        r"\label{tab:bias_results}",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{\extracolsep{\fill}}l" + "c" * n + "}",
        r"\hline",
        "Demographic & " + " & ".join(f"\\textsc{{{m}}}" for m in models) + r" \\",
        r"\hline",
    ]
    
    # Men and Women sections
    for g, label in [('M', 'Men'), ('F', 'Women')]:
        # Overall row
        vals = [all_stats[m][f'mean_{g}'] for m in models]
        lines.append(f"{label} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
        
        # Race rows with min/max formatting
        for race in RACES:
            row = []
            for m in models:
                race_vals = [all_stats[m][f'mean_{g}_{r}'] for r in RACES]
                val = all_stats[m][f'mean_{g}_{race}']
                row.append(format_val(val, val == min(race_vals), val == max(race_vals)))
            lines.append(f"\\quad {RACE_SHORT[race]} & " + " & ".join(row) + r" \\")
        
        lines.append(r"\hline")
    
    # P-value rows
    lines.append(r"\quad Sex & " + " & ".join(format_p(all_stats[m]['p_sex']) for m in models) + r" \\")
    lines.append(r"\quad Race & " + " & ".join(format_p(all_stats[m]['p_race']) for m in models) + r" \\")
    lines.append(r"\quad Sex \& Race & " + " & ".join(format_p(all_stats[m]['p_combined']) for m in models) + r" \\")
    
    lines.extend([r"\hline", r"\end{tabular}}", r"\end{table*}"])
    
    return "\n".join(lines)


def print_summary(all_stats: dict, load_info: dict):
    """Print comprehensive statistics summary."""
    models = list(all_stats.keys())
    
    print("\n" + "=" * 100)
    print("STATISTICS SUMMARY")
    print("=" * 100)
    
    # Data loading info
    print("\n📊 DATA LOADING INFO")
    print("-" * 80)
    print(f"{'Model':<20} {'Original':>10} {'Dropped':>10} {'Final':>10} {'Pairs':>10}")
    for m in models:
        info = load_info.get(m, {})
        orig = info.get('original_rows', 'N/A')
        dropped = info.get('dropped_na', 0)
        final = info.get('final_rows', all_stats[m]['n'])
        pairs = all_stats[m].get('n_combined_pairs', 'N/A')
        print(f"{m:<20} {orig:>10} {dropped:>10} {final:>10} {pairs:>10}")
    
    # Mean acuity by gender
    print("\n📊 MEAN ACUITY BY GENDER")
    print("-" * 60)
    print(f"{'Model':<20} {'Men':>8} {'Women':>8} {'Δ (W-M)':>10}")
    for m in models:
        men = all_stats[m]['mean_M']
        women = all_stats[m]['mean_F']
        diff = women - men
        sig = '*' if all_stats[m]['p_sex'] < 0.05 else ''
        print(f"{m:<20} {men:>8.3f} {women:>8.3f} {diff:>+10.3f}{sig}")
    
    # Mean acuity by race (averaged across gender)
    print("\n📊 MEAN ACUITY BY RACE (averaged across gender)")
    print("-" * 100)
    header = f"{'Model':<20}" + "".join(f"{RACE_SHORT[r]:>12}" for r in RACES)
    print(header)
    for m in models:
        vals = []
        for r in RACES:
            avg = (all_stats[m][f'mean_M_{r}'] + all_stats[m][f'mean_F_{r}']) / 2
            vals.append(f"{avg:>12.3f}")
        print(f"{m:<20}" + "".join(vals))
    
    # Prediction variance (instability)
    print("\n📊 PREDICTION INSTABILITY (variance per patient)")
    print("-" * 60)
    print(f"{'Model':<20} {'Mean Var':>12} {'Max Var':>12}")
    for m in models:
        mean_var = all_stats[m].get('mean_variance', 0)
        max_var = all_stats[m].get('max_variance', 0)
        print(f"{m:<20} {mean_var:>12.4f} {max_var:>12.4f}")
    
    # Statistical significance - RAW p-values
    print("\n📊 STATISTICAL SIGNIFICANCE - RAW p-values")
    print("-" * 80)
    print(f"{'Model':<20} {'Sex':>12} {'Race':>12} {'Combined':>12}")
    for m in models:
        p_sex = all_stats[m]['p_sex']
        p_race = all_stats[m]['p_race']
        p_comb = all_stats[m]['p_combined']
        sex_str = f"{p_sex:.4f}" + ("**" if p_sex < 0.01 else ("*" if p_sex < 0.05 else ""))
        race_str = f"{p_race:.4f}" + ("**" if p_race < 0.01 else ("*" if p_race < 0.05 else ""))
        comb_str = f"{p_comb:.4f}" + ("**" if p_comb < 0.01 else ("*" if p_comb < 0.05 else ""))
        print(f"{m:<20} {sex_str:>12} {race_str:>12} {comb_str:>12}")
    
    # Statistical significance - BONFERRONI-ADJUSTED p-values
    print("\n📊 STATISTICAL SIGNIFICANCE - BONFERRONI-ADJUSTED p-values (×3)")
    print("-" * 80)
    print(f"{'Model':<20} {'Sex':>12} {'Race':>12} {'Combined':>12}")
    for m in models:
        p_sex = all_stats[m]['p_sex_adj']
        p_race = all_stats[m]['p_race_adj']
        p_comb = all_stats[m]['p_combined_adj']
        sex_str = f"{p_sex:.4f}" + ("**" if p_sex < 0.01 else ("*" if p_sex < 0.05 else ""))
        race_str = f"{p_race:.4f}" + ("**" if p_race < 0.01 else ("*" if p_race < 0.05 else ""))
        comb_str = f"{p_comb:.4f}" + ("**" if p_comb < 0.01 else ("*" if p_comb < 0.05 else ""))
        print(f"{m:<20} {sex_str:>12} {race_str:>12} {comb_str:>12}")
    
    # Bias detection summary (using adjusted p-values)
    print("\n📊 BIAS DETECTION SUMMARY (Bonferroni-adjusted, α=0.05)")
    print("-" * 60)
    n_sex = sum(1 for m in models if all_stats[m]['p_sex_adj'] < 0.05)
    n_race = sum(1 for m in models if all_stats[m]['p_race_adj'] < 0.05)
    n_comb = sum(1 for m in models if all_stats[m]['p_combined_adj'] < 0.05)
    print(f"  Models with significant SEX bias:      {n_sex}/{len(models)}")
    print(f"  Models with significant RACE bias:     {n_race}/{len(models)}")
    print(f"  Models with significant COMBINED bias: {n_comb}/{len(models)}")
    
    # Identify most/least biased models
    unbiased = [m for m in models if all_stats[m]['p_sex_adj'] >= 0.05 and 
                all_stats[m]['p_race_adj'] >= 0.05 and all_stats[m]['p_combined_adj'] >= 0.05]
    if unbiased:
        print(f"\n  ✅ Models with NO significant bias: {', '.join(unbiased)}")
    
    biased_all = [m for m in models if all_stats[m]['p_sex_adj'] < 0.05 and 
                  all_stats[m]['p_race_adj'] < 0.05 and all_stats[m]['p_combined_adj'] < 0.05]
    if biased_all:
        print(f"  ⚠️  Models with ALL significant biases: {', '.join(biased_all)}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze demographic bias in triage predictions')
    parser.add_argument('--output-tex', type=str, help='Save LaTeX table to file')
    parser.add_argument('--results-dir', type=str, default=str(RESULTS_DIR),
                        help='Directory containing result CSVs')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load all models
    print("=" * 100)
    print("LOADING DATA")
    print("=" * 100)
    
    all_stats = {}
    load_info = {}
    for model_name, filename in MODELS:
        filepath = results_dir / filename
        if not filepath.exists():
            print(f"⚠️  Skipping {model_name}: file not found")
            continue
        
        print(f"Loading {model_name}...")
        df, info = load_data(str(filepath))
        all_stats[model_name] = compute_stats(df)
        load_info[model_name] = info
        
        if info['dropped_na'] > 0:
            print(f"   ⚠️  Dropped {info['dropped_na']} rows with missing Estimated_Acuity")
    
    if not all_stats:
        print("❌ No data loaded!")
        return
    
    # Generate and print LaTeX table
    latex = generate_latex(all_stats)
    print("\n" + "=" * 100)
    print("LATEX TABLE")
    print("=" * 100)
    print(latex)
    
    if args.output_tex:
        with open(args.output_tex, 'w') as f:
            f.write(latex)
        print(f"\n✅ LaTeX saved to: {args.output_tex}")
    
    # Print summary statistics
    print_summary(all_stats, load_info)


if __name__ == '__main__':
    main()