from typing import List, Dict, Set, Tuple
import pandas as pd
import re
from collections import Counter
import spacy
from datetime import datetime
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PARTIES = [
    "cdu", "csu", "spd", "grüne", "dielinke", "fdp", "afd", "bsw"]

custom_colors_topics = [
    '#CEDB2D',  # ZEW GRÜN - Bright lime for Fiskal/Monetär
    '#A9A820',  # TEXT-/LINIENGRÜN - Olive for Arbeit/Soziales
    '#8B4513',  # Saddle Brown for Marktwettbewerb (distinctive brown)
    '#2E8B57',  # Sea Green for Umwelt & Energie (more vivid green)
    '#1A5276',  # Strong Blue for Digitale Infrastruktur (distinctive blue)
    '#7FBC8F',  # Medium Sea Green for Industrie (lighter green)
    '#E2EFF2',  # ZEW EISBLAU - Light blue for Finanzmärkte
    '#9370DB',  # Medium Purple for Wohnen & Immobilien (purple tone)
    '#4a4a49',  # ZEW GRAU - Dark gray for Öffentliche Politik
]

# Combine both term lists while avoiding duplicates
COMBINED_ECONOMIC_TERMS = set([
    # Original comprehensive economic terms
    "wirtschaft", "ökonomie", "finanzen", "haushalt", "steuern", "staatsschulden",
    "zinsen", "inflation", "deflation", "rezession", "konjunktur", "wachstum",
    "bruttoinlandsprodukt", "bip", "arbeitsmarkt", "arbeitslosigkeit", "beschäftigung",
    "gehalt", "lohn", "mindestlohn", "rente", "fachkräftemangel", "sozialversicherung",
    "gesundheitssystem", "krankenversicherung", "pflegeversicherung", "hartz iv",
    "sozialhilfe", "grundsicherung", "investitionen", "subventionen", "fördermittel",
    "exportwirtschaft", "außenhandel", "handelsbilanz", "import", "export",
    "freihandel", "protektionismus", "globalisierung", "handelsabkommen", "zölle",
    "binnenmarkt", "wettbewerb", "monopol", "kartell", "fusionen", "übernahmen",
    "mittelstand", "startup", "gründungen", "insolvenz", "banken", "sparkassen",
    "kredite", "darlehen", "anleihen", "aktien", "börse", "fonds", "versicherungen",
    "altersvorsorge", "vermögen", "immobilien", "mieten", "wohnungsbau",
    "grundsteuer", "erbschaftssteuer", "mehrwertsteuer", "ökosteuer", "energiepreise",
    "strompreise", "benzinpreise", "dieselpreise", "ölpreis", "kohleausstieg",
    "energiewende", "umweltschutz", "klimawandel", "emissionen", "co2-steuer",
    "digitalisierung", "breitbandausbau", "cybersicherheit", "ki", "industrie 4.0",
    "infrastruktur", "verkehrspolitik", "maut", "pendlerpauschale", "dieselskandal",
    "elektromobilität", "batterieforschung", "stahl", "chemie", "maschinenbau",
    "automobilindustrie", "landwirtschaft", "agrarsubventionen", "dürrehilfen",
    "milchpreise", "fleischproduktion", "tierschutz", "verbraucherschutz",
    "produktsicherheit", "lebensmittelstandards", "gentechnik", "glyphosat",
    
    # Political Economic Policies
    "schuldenbremse", "bundeshaushalt", "länderfinanzausgleich", "konjunkturpaket", 
    "wirtschaftsförderung", "strukturförderung", "steuerreform", "steuerpolitik", 
    "vermögensteuer", "sozialpolitik", "arbeitsmarktpolitik",
    
    # Political Institutions
    "bundesbank", "finanzministerium", "wirtschaftsministerium", "kartellamt", 
    "bundesnetzagentur", "finanzaufsicht", "rechnungshof", "wirtschaftsausschuss", 
    "finanzausschuss",
    
    # Reform Terms
    "rentenreform", "steuerreform", "gesundheitsreform", 
    "verkehrswende", "mobilitätswende", "bürokratieabbau", "verwaltungsreform",
    
    # Political Debates
    "schuldenpolitik", "sparpolitik", "austerität", "industriepolitik", 
    "standortpolitik", "mittelstandspolitik", "verteilungsgerechtigkeit", 
    "steuergerechtigkeit", "wirtschaftssanktionen", "handelspolitik",
    
    # Current Issues
    "bürgergeld", "grundrente", "kindergrundsicherung", "mietpreisbremse", 
    "mietendeckel", "wohnungspolitik", "arbeitsmigration", 
    "einwanderungsgesetz",
    
    # Party-Related
    "wirtschaftsflügel", "wirtschaftsrat", "wirtschaftsprogramm", "parteifinanzen", 
    "parteispenden", "wahlkampffinanzierung", "koalitionsvertrag", 
    "regierungsprogramm",
    
    # Budget Terms
    "haushaltsberatungen", "haushaltssperre", "nachtragshaushalt", "staatsausgaben", 
    "staatsverschuldung", "finanzplanung", "investitionsprogramm",
    
    # EU/International
    "europäische zentralbank", "ezb", "währungsunion", "stabilitätspakt", 
    "fiskalpakt", "wiederaufbaufonds", "handelsabkommen",
    
    # Additional Political Economic Terms
    "schwarze null", "marktwirtschaft", "soziale marktwirtschaft", "planwirtschaft",
    "vermögensabgabe", "reichensteuer", "finanztransaktionssteuer",
    
    # Regional Politics
    "länderhaushalt", "kommunalfinanzen", "städtebauförderung", "strukturwandel",
    "regionalförderung", "kommunalhaushalt",

    #Additional terms
    "lieferkettengesetz", "steuersenkung", "steuererhöhung", "steuerhinterziehung",
    "kaufkraft", "lieferkettengesetz", "schattenhaushalt", "arbeitskräftemangel",
    "binnenkonjunktur", "wirtschaftswachstum", "modernisierungsumlage", "sozialleistung",
    "erbschaftssteuer", "bürokratie", "kapitalismus", "sozialstaat"
])

class EnhancedPoliticalEconomicFilter:
    def __init__(self):
        self.terms = COMBINED_ECONOMIC_TERMS
        self.preprocessor = EconomicContentPreprocessor()
        
        # Enhanced categorization system
        self.categories = {
            'fiscal_monetary': [
                'haushalt', 'steuer', 'schulden', 'finanz', 'währung', 'inflation',
                'zinsen', 'geldpolitik', 'bundesbank', 'ezb'
            ],
            'labor_social': [
                'arbeit', 'beschäftigung', 'sozial', 'rente', 'versicherung',
                'mindestlohn', 'gehalt', 'lohn', 'fachkräfte', 'bürgergeld'
            ],
            'market_competition': [
                'wettbewerb', 'markt', 'handel', 'export', 'import', 'binnenmarkt',
                'kartell', 'monopol', 'fusion', 'mittelstand'
            ],
            'environment_energy': [
                'klima', 'umwelt', 'energie', 'emission', 'co2', 'kohle',
                'erneuerbar', 'öko', 'elektro'
            ],
            'infrastructure_digital': [
                'infrastruktur', 'digital', 'breitband', 'verkehr', 'mobil',
                'cyber', 'ki', 'industrie 4.0'
            ],
            'industry_sectors': [
                'industrie', 'automobil', 'maschinenbau', 'chemie', 'landwirtschaft',
                'produktion', 'gewerbe', 'handwerk'
            ],
            'financial_markets': [
                'börse', 'aktien', 'fonds', 'bank', 'kredit', 'anleihe',
                'investition', 'vermögen'
            ],
            'housing_property': [
                'immobilien', 'miete', 'wohnung', 'bau', 'grundstück',
                'eigentum', 'mietpreis'
            ],
            'public_policy': [
                'politik', 'reform', 'gesetz', 'regulierung', 'förderung',
                'subvention', 'programm'
            ]
        }
        
        # Create variations of terms
        self.term_variations = self._generate_term_variations()
        
        # Create regex pattern
        terms_pattern = '|'.join(r'\b{}\b'.format(re.escape(term)) 
                                for term in self.term_variations)
        self.terms_regex = re.compile(terms_pattern, re.IGNORECASE)

    def _generate_term_variations(self) -> Set[str]:
        """Generate variations of terms including compounds and common forms"""
        variations = set()
        
        for term in self.terms:
            # Add original term
            variations.add(term)
            
            # Add hyphenated and compound variations
            if ' ' in term:
                variations.add(term.replace(' ', '-'))
                variations.add(term.replace(' ', ''))
            
            # Add common endings for German compounds
            endings = ['politik', 'system', 'markt', 'förderung', 'entwicklung']
            for ending in endings:
                if not term.endswith(ending):
                    variations.add(f"{term}{ending}")
        
        return variations
    
    def filter_posts(self, df: pd.DataFrame, text_column: str = 'text') -> Tuple[pd.DataFrame, Dict]:
        """
        Filter and analyze posts for political economic content.
        
        Args:
            df (pd.DataFrame): DataFrame containing posts
            text_column (str): Name of the column containing post text
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Filtered DataFrame and analysis results
        """
        # Create a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Preprocess texts
        filtered_df['preprocessed_text'] = filtered_df[text_column].apply(
            self.preprocessor.preprocess_text
        )
        
        # Add analysis columns
        filtered_df['contains_economic_terms'] = filtered_df['preprocessed_text'].apply(
            lambda x: bool(self.terms_regex.search(str(x)))
        )
        
        filtered_df['economic_terms_found'] = filtered_df['preprocessed_text'].apply(
            self.extract_economic_terms
        )
        
        filtered_df['economic_terms_count'] = filtered_df['economic_terms_found'].apply(len)
        
        # Generate analysis
        analysis = self.analyze_political_economic_content(filtered_df)
        
        # Filter only posts with economic terms
        economic_posts = filtered_df[filtered_df['contains_economic_terms']]
        
        return economic_posts, analysis

    def extract_economic_terms(self, text: str) -> List[str]:
        """Extract economic terms from preprocessed text"""
        if not isinstance(text, str):
            return []
        
        found_terms = []
        for term in self.term_variations:
            if re.search(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                found_terms.append(term)
        
        return list(set(found_terms))

    def analyze_political_economic_content(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis of political economic content"""
        economic_posts = df[df['contains_economic_terms']]
        
        # Flatten all found terms
        all_terms = [
            term 
            for terms_list in economic_posts['economic_terms_found'].tolist() 
            for term in terms_list
        ]
        
        term_counts = Counter(all_terms)
        
        # Calculate term co-occurrences
        co_occurrences = Counter()
        for terms_list in economic_posts['economic_terms_found']:
            if len(terms_list) > 1:
                for i, term1 in enumerate(terms_list):
                    for term2 in terms_list[i+1:]:
                        if term1 < term2:
                            co_occurrences[(term1, term2)] += 1
                        else:
                            co_occurrences[(term2, term1)] += 1
        
        # Categorize terms
        categorized_terms = {}
        for category, keywords in self.categories.items():
            category_terms = []
            for term, count in term_counts.items():
                if any(keyword in term.lower() for keyword in keywords):
                    category_terms.append({'term': term, 'count': count})
            categorized_terms[category] = category_terms
        
        # Time-based analysis
        daily_counts = {}
        if 'timestamp' in economic_posts.columns:
            economic_posts['date'] = pd.to_datetime(economic_posts['timestamp']).dt.strftime('%Y-%m-%d')
            daily_counts = economic_posts.groupby('date').size().to_dict()
        
        analysis = {
            'summary': {
                'total_posts': len(df),
                'economic_posts': len(economic_posts),
                'percentage_economic': (len(economic_posts) / len(df) * 100) if len(df) > 0 else 0,
                'unique_terms_used': len(term_counts),
                'average_terms_per_post': len(all_terms) / len(economic_posts) if len(economic_posts) > 0 else 0
            },
            'term_frequency': dict(term_counts),
            'most_common_terms': dict(term_counts.most_common(20)),
            'common_co_occurrences': {f"{t1}-{t2}": count 
                                    for (t1, t2), count in co_occurrences.most_common(10)},
            'daily_post_counts': daily_counts,
            'categories': categorized_terms
        }
        
        return analysis

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for term extraction"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Handle hashtags - keep the text without #
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Convert compound words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        return text



class EconomicContentPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with German language model"""
        self.nlp = spacy.load('de_core_news_sm')
        
        # Add special cases for economic terms
        self.special_cases = {
            "hartz4": "hartz iv",
            "h4": "hartz iv",
            "alg2": "arbeitslosengeld ii",
            "alg1": "arbeitslosengeld",
            "mwst": "mehrwertsteuer",
            "est": "einkommensteuer",
            "kv": "krankenversicherung",
            "rv": "rentenversicherung",
            "pv": "pflegeversicherung",
            "bip": "bruttoinlandsprodukt",
            "ezb": "europäische zentralbank",
            "eu": "europäische union"
        }
        
        # Common economic hashtag prefixes/suffixes
        self.hashtag_patterns = [
            "wirtschaft", "öko", "finanz", "steuer", "geld", "markt", 
            "handel", "börse", "bank", "versicherung", "inflation"
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for economic term analysis.
        
        Args:
            text (str): Raw text from post
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle hashtags
        text = self._process_hashtags(text)
        
        # Replace special cases
        for abbrev, full in self.special_cases.items():
            text = re.sub(rf'\b{abbrev}\b', full, text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Lemmatize and keep relevant parts of speech
        tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct 
            and token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']
        ]
        
        return ' '.join(tokens)
    
    def _process_hashtags(self, text: str) -> str:
        """Process hashtags to extract economic terms"""
        # Find all hashtags
        hashtags = re.findall(r'#(\w+)', text)
        
        for hashtag in hashtags:
            # Convert camelCase to spaces
            expanded = re.sub(r'([a-z])([A-Z])', r'\1 \2', hashtag).lower()
            
            # Check if hashtag contains economic patterns
            if any(pattern in expanded for pattern in self.hashtag_patterns):
                # Add expanded version to text
                text += f" {expanded}"
        
        return text
    
    def find_party_hashtags(self, text: str) -> List[str]:
        """Find party-related hashtags in the text"""
        # Find all hashtags
        hashtags = re.findall(r'#(\w+)', text)
        
        # Check if hashtag contains party-related terms
        party_hashtags = [
            hashtag for hashtag in hashtags 
            if any(party in hashtag.lower() for party in PARTIES)
        ]
        
        return party_hashtags

class EnhancedEconomicContentFilter:
    def __init__(self, economic_terms: List[str]):
        """
        Initialize the enhanced filter with preprocessing capabilities.
        
        Args:
            economic_terms (List[str]): List of economic terms to filter for
        """
        self.economic_terms = set(term.lower() for term in economic_terms)
        self.preprocessor = EconomicContentPreprocessor()
        
        # Create variations of terms
        self.term_variations = self._generate_term_variations()
        
        # Create regex pattern
        terms_pattern = '|'.join(r'\b{}\b'.format(re.escape(term)) for term in self.term_variations)
        self.terms_regex = re.compile(terms_pattern, re.IGNORECASE)
    
    def _generate_term_variations(self) -> Set[str]:
        """Generate common variations of economic terms"""
        variations = set()
        
        for term in self.economic_terms:
            # Add original term
            variations.add(term)
            
            # Add compound variations
            if ' ' in term:
                # Add hyphenated version
                variations.add(term.replace(' ', '-'))
                # Add concatenated version
                variations.add(term.replace(' ', ''))
            
            # Add common abbreviations
            words = term.split()
            if len(words) > 1:
                # Add acronym
                acronym = ''.join(word[0] for word in words)
                variations.add(acronym)
        
        return variations
    
    def _find_original_term(self, variation: str) -> str:
        """Find the original term for a given variation"""
        # First check if it's an original term
        if variation in self.economic_terms:
            return variation
            
        # Check variations of each original term
        for original_term in self.economic_terms:
            term_variations = {
                original_term,
                original_term.replace(' ', '-'),
                original_term.replace(' ', ''),
                ''.join(word[0] for word in original_term.split())
            }
            if variation in term_variations:
                return original_term
                
        return variation
    
    def filter_posts(self, df: pd.DataFrame, text_column: str = 'text') -> Tuple[pd.DataFrame, Dict]:
        """
        Filter and analyze posts for economic content.
        
        Args:
            df (pd.DataFrame): DataFrame containing posts
            text_column (str): Name of the column containing post text
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Filtered DataFrame and analysis results
        """
        # Create a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Preprocess texts
        filtered_df['preprocessed_text'] = filtered_df[text_column].apply(
            self.preprocessor.preprocess_text
        )

        filtered_df["party_hashtags"] = filtered_df[text_column].apply(
            self.preprocessor.find_party_hashtags
        )
        
        # Add analysis columns
        filtered_df['contains_economic_terms'] = filtered_df['preprocessed_text'].apply(
            lambda x: bool(self.terms_regex.search(str(x)))
        )
        
        filtered_df['economic_terms_found'] = filtered_df['preprocessed_text'].apply(
            self.extract_economic_terms
        )
        
        filtered_df['economic_terms_count'] = filtered_df['economic_terms_found'].apply(len)
        
        # Generate analysis
        analysis = self.analyze_economic_content(filtered_df)
        
        # Filter only posts with economic terms
        economic_posts = filtered_df[filtered_df['contains_economic_terms']]
        
        return economic_posts, analysis
    
    def extract_economic_terms(self, text: str) -> List[str]:
        """Extract economic terms from preprocessed text"""
        if not isinstance(text, str):
            return []
        
        found_terms = []
        for term in self.term_variations:
            if re.search(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                original_term = self._find_original_term(term)
                found_terms.append(original_term)
        
        return list(set(found_terms))
    
    def analyze_economic_content(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis of economic content"""
        economic_posts = df[df['contains_economic_terms']].copy()  # Create explicit copy
        
        # Flatten all found terms
        all_terms = [
            term 
            for terms_list in economic_posts['economic_terms_found'].tolist() 
            for term in terms_list
        ]
        
        term_counts = Counter(all_terms)
        
        # Calculate term co-occurrences
        co_occurrences = Counter()
        for terms_list in economic_posts['economic_terms_found']:
            if len(terms_list) > 1:
                for i, term1 in enumerate(terms_list):
                    for term2 in terms_list[i+1:]:
                        if term1 < term2:
                            co_occurrences[(term1, term2)] += 1
                        else:
                            co_occurrences[(term2, term1)] += 1
        
        # Time-based analysis
        daily_counts = {}
        if 'timestamp' in economic_posts.columns:
            economic_posts['date'] = pd.to_datetime(economic_posts['timestamp'])
            daily_counts = economic_posts.groupby(economic_posts['date'].dt.strftime('%Y-%m-%d')).size().to_dict()
        
        analysis = {
            'summary': {
                'total_posts': len(df),
                'economic_posts': len(economic_posts),
                'percentage_economic': (len(economic_posts) / len(df) * 100) if len(df) > 0 else 0,
                'unique_terms_used': len(term_counts),
                'average_terms_per_post': len(all_terms) / len(economic_posts) if len(economic_posts) > 0 else 0
            },
            'term_frequency': dict(term_counts),
            'most_common_terms': dict(term_counts.most_common(20)),
            'common_co_occurrences': {f"{t1}-{t2}": count 
                                    for (t1, t2), count in co_occurrences.most_common(10)},
            'daily_post_counts': daily_counts,
            'term_categories': self.categorize_terms(term_counts)
        }
        
        return analysis
    
    def categorize_terms(self, term_counts: Counter) -> Dict:
        """Categorize found terms into economic subtopics"""
        categories = {
            'monetary': ['inflation', 'zinsen', 'währung', 'geld', 'euro'],
            'fiscal': ['steuern', 'haushalt', 'schulden', 'ausgaben'],
            'labor': ['arbeit', 'beschäftigung', 'gehalt', 'lohn', 'arbeitslos'],
            'market': ['handel', 'markt', 'wettbewerb', 'export', 'import'],
            'social': ['rente', 'versicherung', 'sozial', 'gesundheit']
        }
        
        categorized = {category: [] for category in categories}
        
        for term, count in term_counts.items():
            for category, keywords in categories.items():
                if any(keyword in term for keyword in keywords):
                    categorized[category].append({'term': term, 'count': count})
        
        return categorized
    
    def co_occurrences_parties_terms(self, df: pd.DataFrame) -> Dict:
        """Find co-occurrences between party hashtags and economic terms"""
        party_terms = {party: Counter() for party in PARTIES}
        
        for _, row in df.iterrows():
            for party in row['party_hashtags']:
                for term in row['economic_terms_found']:
                    party_terms[party][term] += 1
        
        return party_terms

def main():
    """Main function to run political economic analysis on Bluesky posts"""
    
    # Initialize paths
    base_dir = 'H:/bluesky_data'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, 'political_economic_analysis', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize filter
    economic_filter = EnhancedPoliticalEconomicFilter()
    
    try:
        
        # Load posts
        print("Loading posts...")
        posts_df = pd.read_csv(os.path.join(base_dir, 'analysis/unique_posts_20250311_093551.csv'))
        """
        # Filter and analyze posts
        print("Analyzing posts...")
        economic_posts, analysis = economic_filter.filter_posts(posts_df)
        
        # Save filtered posts
        print("Saving filtered posts...")
        economic_posts.to_csv(os.path.join(output_dir, 'political_economic_posts.csv'), index=False)

        # Save analysis results
        print("Saving analysis results...")
        with open(os.path.join(output_dir, 'political_economic_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        """
        economic_posts = pd.read_csv(os.path.join("H:/bluesky_data/political_economic_analysis/20250311_131554", 'political_economic_posts.csv'))

        analysis = json.load(open(os.path.join("H:/bluesky_data/political_economic_analysis/20250311_131554", 'political_economic_analysis.json')))
        
        # Generate visualizations
        print("Generating visualizations...")
        generate_visualizations(economic_posts, analysis, output_dir)

        category_results = generate_category_pie_charts(economic_posts, output_dir)
        
        # Print summary
        print_analysis_summary(analysis)

        stats = analyze_economic_percentage_over_time(posts_df, economic_posts, output_dir)

        calculate_election_period_stats(posts_df, economic_posts, output_dir)
        
        print(f"\nAnalysis complete. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

def generate_category_pie_charts(df, output_dir, election_date='2025-02-23'):
    """
    Generate pie charts showing distribution of economic term categories 
    before and after the election date
    
    Args:
        df (pd.DataFrame): DataFrame containing economic posts with economic_terms_found
        output_dir (str): Directory to save the charts
        election_date (str): Election date in YYYY-MM-DD format
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert date columns to datetime
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    election_date = pd.to_datetime(election_date).date()
    
    # Filter to relevant date range (Jan 23 to Mar 10)
    df_filtered = df[(df['date'] >= pd.to_datetime('2025-01-23').date()) & 
                     (df['date'] <= pd.to_datetime('2025-03-10').date())]
    
    # Split into before and after election
    df_before = df_filtered[df_filtered['date'] < election_date]
    df_after = df_filtered[df_filtered['date'] >= election_date]
    
    print(f"Total posts in timeframe: {len(df_filtered)}")
    print(f"Posts before election: {len(df_before)}")
    print(f"Posts after election: {len(df_after)}")
    
    # Define categories and their German labels
    categories = {
        'fiscal_monetary': 'Fiskal/Monetär',
        'labor_social': 'Arbeit/Soziales',
        'market_competition': 'Marktwettbewerb',
        'environment_energy': 'Umwelt & Energie',
        'infrastructure_digital': 'Digitale Infrastruktur',
        'industry_sectors': 'Industrie',
        'financial_markets': 'Finanzmärkte',
        'housing_property': 'Wohnen & Immobilien',
        'public_policy': 'Öffentliche Politik'
    }
    
    # Define keywords for categorizing terms
    category_keywords = {
        'fiscal_monetary': [
            'haushalt', 'steuer', 'schulden', 'finanz', 'währung', 'inflation',
            'zinsen', 'geldpolitik', 'bundesbank', 'ezb'
        ],
        'labor_social': [
            'arbeit', 'beschäftigung', 'sozial', 'rente', 'versicherung',
            'mindestlohn', 'gehalt', 'lohn', 'fachkräfte', 'bürgergeld'
        ],
        'market_competition': [
            'wettbewerb', 'markt', 'handel', 'export', 'import', 'binnenmarkt',
            'kartell', 'monopol', 'fusion', 'mittelstand'
        ],
        'environment_energy': [
            'klima', 'umwelt', 'energie', 'emission', 'co2', 'kohle',
            'erneuerbar', 'öko', 'elektro'
        ],
        'infrastructure_digital': [
            'infrastruktur', 'digital', 'breitband', 'verkehr', 'mobil',
            'cyber', 'ki', 'industrie 4.0'
        ],
        'industry_sectors': [
            'industrie', 'automobil', 'maschinenbau', 'chemie', 'landwirtschaft',
            'produktion', 'gewerbe', 'handwerk'
        ],
        'financial_markets': [
            'börse', 'aktien', 'fonds', 'bank', 'kredit', 'anleihe',
            'investition', 'vermögen'
        ],
        'housing_property': [
            'immobilien', 'miete', 'wohnung', 'bau', 'grundstück',
            'eigentum', 'mietpreis'
        ],
        'public_policy': [
            'politik', 'reform', 'gesetz', 'regulierung', 'förderung',
            'subvention', 'programm'
        ]
    }
    
    # Function to categorize terms and count them
    def categorize_and_count_terms(dataframe):
        # Initialize counts for each category
        category_counts = {category: 0 for category in categories.keys()}
        
        # Iterate through each post
        for _, row in dataframe.iterrows():
            terms = row['economic_terms_found']
            
            # Handle different formats of economic_terms_found
            if isinstance(terms, str):
                try:
                    # Try to evaluate if it's a string representation of a list
                    terms = eval(terms)
                except:
                    # If eval fails, split by comma (simple approach)
                    terms = terms.split(',')
            
            # Skip if terms is not iterable
            if not isinstance(terms, (list, tuple, set)):
                continue
                
            # Count each term in appropriate categories
            for term in terms:
                term = str(term).lower()
                for category, keywords in category_keywords.items():
                    if any(keyword in term for keyword in keywords):
                        category_counts[category] += 1
                        break  # Assign to first matching category
        
        return category_counts
    
    # Count terms for before and after election
    before_counts = categorize_and_count_terms(df_before)
    after_counts = categorize_and_count_terms(df_after)
    
    print("Category counts before election:", before_counts)
    print("Category counts after election:", after_counts)
    
    # Generate pie charts
    def create_pie_chart(counts, title, filename):
        # Check if all values are zero
        if sum(counts.values()) == 0:
            print(f"Warning: All category counts are zero for {title}")
            plt.figure(figsize=(10, 8))
            plt.text(0.5, 0.5, "No data available for this period", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.axis('off')
            plt.savefig(filename, dpi=300)
            plt.close()
            return
        
        # Create the pie chart
        plt.figure(figsize=(12, 10))
        
        # Get labels and values, filtering out zero values
        labels = []
        values = []
        for cat, count in counts.items():
            if count > 0:
                labels.append(categories[cat])
                values.append(count)
        
        # Choose a colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        
        # Create pie chart
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=custom_colors_topics)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title(title, fontsize=16)
        
        # Add legend with percentages
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    # Create and save pie charts
    create_pie_chart(
        before_counts, 
        'Verteilung der Wirtschaftsbegriffe vor der Bundestagswahl (23.01. - 22.02.2025)',
        os.path.join(output_dir, 'economic_categories_before_election.png')
    )
    
    create_pie_chart(
        after_counts, 
        'Verteilung der Wirtschaftsbegriffe nach der Bundestagswahl (23.02. - 10.03.2025)',
        os.path.join(output_dir, 'economic_categories_after_election.png')
    )
    
    # Calculate and visualize changes between periods
    def visualize_category_changes(before_counts, after_counts):
        categories_to_show = [cat for cat, count in before_counts.items() if before_counts[cat] > 0 or after_counts[cat] > 0]
        
        if not categories_to_show:
            print("No categories with non-zero counts to visualize changes")
            return
            
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Calculate percentages for better comparison
        before_total = sum(before_counts.values())
        after_total = sum(after_counts.values())
        
        before_pct = {cat: count/before_total*100 if before_total > 0 else 0 
                     for cat, count in before_counts.items()}
        after_pct = {cat: count/after_total*100 if after_total > 0 else 0 
                    for cat, count in after_counts.items()}
        
        # Calculate percentage point changes
        pct_changes = {cat: after_pct[cat] - before_pct[cat] for cat in categories_to_show}
        
        # Sort categories by absolute change
        sorted_cats = sorted(categories_to_show, key=lambda x: abs(pct_changes[x]), reverse=True)
        
        # Plot
        x = np.arange(len(sorted_cats))
        width = 0.35
        
        # Get German category labels
        german_labels = [categories[cat] for cat in sorted_cats]
        
        # Create the bars
        before_bars = ax.bar(x - width/2, [before_pct[cat] for cat in sorted_cats], width,
                           label='Vor der Wahl (23.01. - 22.02.)', color='#CEDB2D')
        after_bars = ax.bar(x + width/2, [after_pct[cat] for cat in sorted_cats], width,
                          label='Nach der Wahl (24.02. - 10.03.)', color='#4a4a49')
        
        # Customize the plot
        ax.set_ylabel('Prozentanteil (%)', fontsize=12)
        ax.set_title('Wirtschaftsbegriffskategorien vor und nach der Wahl', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(german_labels, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'economic_categories_comparison.png'), dpi=300)
        plt.close()
    
    # Create the comparison visualization
    visualize_category_changes(before_counts, after_counts)
    
    print(f"Pie charts and comparison chart saved to {output_dir}")
    
    return {
        'before_counts': before_counts,
        'after_counts': after_counts
    }

def generate_visualizations(economic_posts: pd.DataFrame, analysis: Dict, output_dir: str):
    """Generate visualizations for the analysis"""

    print("\nGenerating visualizations...")
    
    # 1. Term frequency plot
    plt.figure(figsize=(15, 8))
    terms = list(analysis['most_common_terms'].keys())[:20]
    counts = [analysis['most_common_terms'][term] for term in terms]
    
    plt.barh(terms, counts)
    plt.title('Top 20 Economic Terms')
    plt.xlabel('Frequency')
    plt.ylabel('Terms')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'term_frequency.png'))
    plt.close()
    
    # 2. Category distribution
    plt.figure(figsize=(12, 8))
    categories = []
    category_counts = []
    
    for category, terms in analysis['categories'].items():
        categories.append(category.replace('_', ' ').title())
        category_counts.append(sum(term['count'] for term in terms))

    plt.pie(category_counts, labels=categories, autopct='%1.1f%%')
    plt.title('Distribution of Economic Terms by Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_distribution_total.png'))
    plt.close()

    window_sizes = [1, 3, 7, 14]  # Generate visualizations for these window sizes
    economic_content_over_time_windows(economic_posts, analysis, output_dir, window_sizes)
    
    # 3. Time series analysis if timestamp available
    if 'timestamp' in economic_posts.columns:
        economic_posts['date'] = pd.to_datetime(economic_posts['timestamp']).dt.date
        # filter out posts earlier than specific date
        economic_posts = economic_posts[economic_posts['date'] >= pd.to_datetime('2025-01-23').date()]
        daily_counts = economic_posts.groupby('date').size()
        
        plt.figure(figsize=(15, 6))
        daily_counts.plot(kind='line', marker='o')
        plt.title('Economic Posts Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Posts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series.png'))
        plt.close()
    
    # 4. Term co-occurrence heatmap
    if 'common_co_occurrences' in analysis:
        co_occurrences = pd.DataFrame(analysis['common_co_occurrences'].items(), 
                                    columns=['pair', 'count'])
        co_occurrences[['term1', 'term2']] = co_occurrences['pair'].str.split('-', expand=True)
        
        # Create matrix
        terms = list(set(co_occurrences['term1'].tolist() + co_occurrences['term2'].tolist()))
        matrix = np.zeros((len(terms), len(terms)))
        term_to_idx = {term: i for i, term in enumerate(terms)}
        
        for _, row in co_occurrences.iterrows():
            i = term_to_idx[row['term1']]
            j = term_to_idx[row['term2']]
            matrix[i, j] = row['count']
            matrix[j, i] = row['count']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, xticklabels=terms, yticklabels=terms, cmap='YlOrRd')
        plt.title('Term Co-occurrence Heatmap')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'co_occurrence_heatmap.png'))
        plt.close()

def print_analysis_summary(analysis: Dict):
    """Print summary of the analysis results"""
    
    print("\nPolitical Economic Analysis Summary")
    print("==================================")
    print(f"Total Posts Analyzed: {analysis['summary']['total_posts']:,}")
    print(f"Posts with Economic Content: {analysis['summary']['economic_posts']:,}")
    print(f"Percentage Economic: {analysis['summary']['percentage_economic']:.2f}%")
    print(f"Average Terms per Post: {analysis['summary']['average_terms_per_post']:.2f}")
    
    print("\nTop 10 Most Common Terms:")
    for term, count in list(analysis['most_common_terms'].items())[:10]:
        print(f"  {term}: {count:,}")
    
    if 'categories' in analysis:
        print("\nCategory Distribution:")
        for category, terms in analysis['categories'].items():
            if terms:
                total = sum(term['count'] for term in terms)
                print(f"\n{category.replace('_', ' ').title()}:")
                print(f"  Total mentions: {total:,}")
                top_terms = sorted(terms, key=lambda x: x['count'], reverse=True)[:3]
                for term in top_terms:
                    print(f"  {term['term']}: {term['count']:,}")
    
    if 'common_co_occurrences' in analysis:
        print("\nTop 5 Term Co-occurrences:")
        for pair, count in list(analysis['common_co_occurrences'].items())[:5]:
            print(f"  {pair}: {count:,}")


# Analysis of economic content over time using stacked chart using the category counts
def economic_content_over_time_windows(economic_posts, analysis, output_dir, window_sizes=[1, 3, 7]):
    """
    Generate stacked bar charts of economic content over time with different window sizes
    
    Args:
        economic_posts (pd.DataFrame): DataFrame containing posts with economic content
        analysis (Dict): Analysis results containing categories
        output_dir (str): Directory to save visualizations
        window_sizes (list): List of window sizes in days to generate plots for
    """
    if 'timestamp' not in economic_posts.columns:
        print("No timestamp column available for time-based analysis.")
        return
    
    # Ensure we have datetime objects
    economic_posts = economic_posts.copy()
    economic_posts['date'] = pd.to_datetime(economic_posts['timestamp']).dt.date

    # Restrict date to posts after specific date
    economic_posts = economic_posts[economic_posts['date'] >= pd.to_datetime('2025-01-23').date()]
    
    # Extract categories from analysis
    categories = list(analysis['categories'].keys())
    
    # Function to count terms by category for each date
    def count_category_terms(df_group):
        # Initialize counts dictionary
        counts = {category: 0 for category in categories}
        
        # Count all terms in this group by category
        for _, row in df_group.iterrows():
            terms = row['economic_terms_found']
            if isinstance(terms, str):
                # If terms are stored as string, convert to list
                try:
                    terms = eval(terms)
                except:
                    terms = []
            
            # Count terms by category
            for term in terms:
                for category, category_terms in analysis['categories'].items():
                    category_term_names = [item['term'] for item in category_terms]
                    if term in category_term_names:
                        counts[category] += 1
        
        return counts
    
    # Group posts by date and calculate category counts
    daily_data = pd.DataFrame()
    for date, group in economic_posts.groupby('date'):
        category_counts = count_category_terms(group)
        row = {'date': date}
        row.update(category_counts)
        daily_data = pd.concat([daily_data, pd.DataFrame([row])], ignore_index=True)
    
    # Sort by date
    daily_data = daily_data.sort_values('date')
    
    # Create plots for each window size
    for window_size in window_sizes:
        print(f"Generating plot for {window_size}-day window...")
        
        # Create a copy of the data for this window
        windowed_data = daily_data.copy()
        
        # Apply rolling window to each category
        if window_size > 1:
            for category in categories:
                windowed_data[category] = windowed_data[category].rolling(
                    window=window_size, min_periods=1
                ).mean()
        
        # Create absolute counts stacked bar chart
        plt.figure(figsize=(15, 8))
        
        # Choose a colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        
        # Create stacked plot
        bottom = np.zeros(len(windowed_data))
        
        for i, category in enumerate(categories):
            plt.bar(
                windowed_data['date'], 
                windowed_data[category],
                bottom=bottom,
                label=category.replace('_', ' ').title(),
                color=custom_colors_topics[i],
                width=0.8  # Adjust bar width as needed
            )
            bottom += windowed_data[category].values
        
        plt.title(f'Economic Content by Category ({window_size}-Day Rolling Average)')
        plt.xlabel('Date')
        plt.ylabel('Number of Term Mentions')
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'economic_content_{window_size}day_window.png'), dpi=300)
        plt.close()
        
        # Calculate percentages for each day
        percentage_data = windowed_data.copy()
        category_totals = percentage_data[categories].sum(axis=1)
        
        for category in categories:
            percentage_data[category] = percentage_data[category] / category_totals * 100
        
        # Create percentage stacked bar chart
        plt.figure(figsize=(15, 8))
        
        # Create stacked percentage plot
        bottom = np.zeros(len(percentage_data))

        category_labels_german = {
            'fiscal_monetary': 'Fiskal/Monetär',
            'labor_social': 'Arbeit/Soziales',
            'market_competition': 'Marktwettbewerb',
            'environment_energy': 'Umwelt & Energie',
            'infrastructure_digital': 'Digitale Infrastruktur',
            'industry_sectors': 'Industrie',
            'financial_markets': 'Finanzmärkte',
            'housing_property': 'Wohnen & Immobilien',
            'public_policy': 'Öffentliche Politik'
        }
        
        for i, category in enumerate(categories):
            plt.bar(
                percentage_data['date'], 
                percentage_data[category],
                bottom=bottom,
                label=category_labels_german[category],
                color=custom_colors_topics[i],
                width=0.8
            )
            bottom += percentage_data[category].values
        
        #plt.title(f'Relative Economic Content by Category ({window_size}-Day Rolling Average)')
        plt.xlabel('Datum')
        plt.ylabel('Prozentualer Anteil der Wirtschaftsbegriffe (%)')
        plt.legend(title='Kategorie', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        # Save the percentage figure
        plt.savefig(os.path.join(output_dir, f'economic_content_pct_{window_size}day_window.png'), dpi=300)
        plt.close()

    print(f"Generated economic content visualizations for {len(window_sizes)} window sizes")

def analyze_economic_percentage_over_time(df, economic_posts, output_dir):
    """
    Analyze and visualize the percentage of economic posts over time
    
    Args:
        df (pd.DataFrame): Original DataFrame with all posts
        economic_posts (pd.DataFrame): DataFrame with only economic posts
        output_dir (str): Directory to save visualizations
    """
    if 'timestamp' not in df.columns:
        print("No timestamp column available for time-based analysis.")
        return
    
    # Ensure we have datetime objects and date column
    df = df.copy()
    economic_posts = economic_posts.copy()
    
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    economic_posts['date'] = pd.to_datetime(economic_posts['timestamp']).dt.date
    
    # Filter posts between two specific dates
    
    df_recent = df[df['date'] >= pd.to_datetime('2025-01-23').date()]
    df_recent = df_recent[df_recent['date'] <= pd.to_datetime('2025-03-10').date()]
    economic_recent = economic_posts[economic_posts['date'] >= pd.to_datetime('2025-01-23').date()]
    economic_recent = economic_recent[economic_recent['date'] <= pd.to_datetime('2025-03-10').date()]
    
    # Count total posts and economic posts per day
    total_by_day = df_recent.groupby('date').size()
    economic_by_day = economic_recent.groupby('date').size()
    
    # Calculate percentage
    percentage_by_day = pd.DataFrame({
        'total': total_by_day,
        'economic': economic_by_day
    }).fillna(0)
    
    percentage_by_day['percentage'] = (percentage_by_day['economic'] / 
                                     percentage_by_day['total'] * 100)
    
    # Create visualizations
    plt.figure(figsize=(15, 8))
    """
    # Plot total posts and economic posts
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(percentage_by_day.index, percentage_by_day['total'], 
            label='Total Posts', marker='o', color='blue')
    ax1.plot(percentage_by_day.index, percentage_by_day['economic'], 
            label='Economic Posts', marker='x', color='green')
    ax1.set_title('Total vs Economic Posts Over Time')
    ax1.set_ylabel('Number of Posts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    """
    # Plot percentage
    ax2 = plt.subplot(1, 1, 1)
    ax2.plot(percentage_by_day.index, percentage_by_day['percentage'], 
            marker='o', color='#CEDB2D')
    #ax2.set_title('Percentage of Posts with Economic Content')
    ax2.set_xlabel('Datum')
    ax2.set_ylabel('Prozent (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add rolling average line (7-day window)
    rolling_avg = percentage_by_day['percentage'].rolling(window=7, min_periods=1).mean()
    ax2.plot(percentage_by_day.index, rolling_avg, 
            label='7-Tages Durchschnitt', linestyle='--', color='#4a4a49')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'economic_content_percentage.png'), dpi=300)
    plt.close()
    
    # Calculate and save statistics
    stats = {
        'overall_percentage': float(len(economic_posts) / len(df) * 100),
        'recent_percentage': float(len(economic_recent) / len(df_recent) * 100),
        'min_daily_percentage': float(percentage_by_day['percentage'].min()),
        'max_daily_percentage': float(percentage_by_day['percentage'].max()),
        'avg_daily_percentage': float(percentage_by_day['percentage'].mean()),
        'median_daily_percentage': float(percentage_by_day['percentage'].median()),
        'trend': 'increasing' if percentage_by_day['percentage'].iloc[-1] > 
                                percentage_by_day['percentage'].iloc[0] else 'decreasing'
    }
    
    # Save statistics as JSON
    with open(os.path.join(output_dir, 'economic_percentage_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats

def calculate_election_period_stats(df, economic_posts, output_dir, election_date='2025-02-23'):
    """
    Calculate statistics for periods before and after the election
    
    Args:
        df (pd.DataFrame): Original DataFrame with all posts
        economic_posts (pd.DataFrame): DataFrame with only economic posts
        output_dir (str): Directory to save visualizations
        election_date (str): Date of the election as string (YYYY-MM-DD)
    """
    if 'timestamp' not in df.columns:
        print("No timestamp column available for time-based analysis.")
        return
    
    # Ensure we have datetime objects and date column
    df = df.copy()
    economic_posts = economic_posts.copy()
    
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    economic_posts['date'] = pd.to_datetime(economic_posts['timestamp']).dt.date
    
    # Convert election date to datetime
    election_date = pd.to_datetime(election_date).date()
    
    # Filter posts between relevant dates
    data_start_date = pd.to_datetime('2025-01-23').date()
    data_end_date = pd.to_datetime('2025-03-10').date()
    
    df_filtered = df[(df['date'] >= data_start_date) & (df['date'] <= data_end_date)]
    economic_filtered = economic_posts[(economic_posts['date'] >= data_start_date) & 
                                      (economic_posts['date'] <= data_end_date)]
    
    # Split into before and after election
    df_before = df_filtered[df_filtered['date'] < election_date]
    df_after = df_filtered[df_filtered['date'] >= election_date]
    
    economic_before = economic_filtered[economic_filtered['date'] < election_date]
    economic_after = economic_filtered[economic_filtered['date'] >= election_date]
    
    # Calculate daily percentages
    def calculate_daily_percentages(df_period, economic_period):
        total_by_day = df_period.groupby('date').size()
        economic_by_day = economic_period.groupby('date').size()
        
        percentage_by_day = pd.DataFrame({
            'total': total_by_day,
            'economic': economic_by_day
        }).fillna(0)
        
        percentage_by_day['percentage'] = (percentage_by_day['economic'] / 
                                         percentage_by_day['total'] * 100)
        
        return percentage_by_day
    
    before_percentages = calculate_daily_percentages(df_before, economic_before)
    after_percentages = calculate_daily_percentages(df_after, economic_after)
    
    # Calculate statistics
    before_stats = {
        'period': 'before_election',
        'date_range': f"{data_start_date} to {election_date - pd.Timedelta(days=1)}",
        'max_percentage': float(before_percentages['percentage'].max()),
        'min_percentage': float(before_percentages['percentage'].min()),
        'avg_percentage': float(before_percentages['percentage'].mean()),
        'median_percentage': float(before_percentages['percentage'].median()),
        'total_posts': int(len(df_before)),
        'economic_posts': int(len(economic_before)),
        'overall_percentage': float(len(economic_before) / len(df_before) * 100) if len(df_before) > 0 else 0
    }
    
    after_stats = {
        'period': 'after_election',
        'date_range': f"{election_date} to {data_end_date}",
        'max_percentage': float(after_percentages['percentage'].max()),
        'min_percentage': float(after_percentages['percentage'].min()),
        'avg_percentage': float(after_percentages['percentage'].mean()),
        'median_percentage': float(after_percentages['percentage'].median()),
        'total_posts': int(len(df_after)),
        'economic_posts': int(len(economic_after)),
        'overall_percentage': float(len(economic_after) / len(df_after) * 100) if len(df_after) > 0 else 0
    }
    
    # Calculate change metrics
    comparison_stats = {
        'percentage_change': {
            'max': (after_stats['max_percentage'] - before_stats['max_percentage']) / before_stats['max_percentage'] * 100,
            'min': (after_stats['min_percentage'] - before_stats['min_percentage']) / before_stats['min_percentage'] * 100,
            'avg': (after_stats['avg_percentage'] - before_stats['avg_percentage']) / before_stats['avg_percentage'] * 100,
            'overall': (after_stats['overall_percentage'] - before_stats['overall_percentage']) / before_stats['overall_percentage'] * 100
        }
    }
    
    # Combine statistics
    election_stats = {
        'before_election': before_stats,
        'after_election': after_stats,
        'comparison': comparison_stats
    }
    
    # Create comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Plot before and after percentages
    ax = plt.subplot(1, 1, 1)
    
    # Calculate days from start for x-axis
    all_percentages = pd.concat([before_percentages, after_percentages])
    
    # Plot with different colors for before and after
    before_line = ax.plot(before_percentages.index, before_percentages['percentage'], 
                         color='blue', marker='o', label='Before Election')
    after_line = ax.plot(after_percentages.index, after_percentages['percentage'], 
                        color='red', marker='o', label='After Election')
    
    # Add election date vertical line
    plt.axvline(x=election_date, color='black', linestyle='--', 
               label=f'Election ({election_date})')
    
    # Add horizontal lines for averages
    plt.axhline(y=before_stats['avg_percentage'], color='blue', linestyle='-.',
               label=f'Before Avg: {before_stats["avg_percentage"]:.2f}%')
    plt.axhline(y=after_stats['avg_percentage'], color='red', linestyle='-.',
               label=f'After Avg: {after_stats["avg_percentage"]:.2f}%')
    
    plt.title('Economic Content Percentage Before and After Election')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'economic_content_election_comparison.png'), dpi=300)
    plt.close()
    
    # Save statistics as JSON
    with open(os.path.join(output_dir, 'election_period_stats.json'), 'w') as f:
        json.dump(election_stats, f, indent=2)
    
    # Print summary
    print("\nElection Period Analysis:")
    print(f"Before Election ({before_stats['date_range']}):")
    print(f"  Max: {before_stats['max_percentage']:.2f}%")
    print(f"  Min: {before_stats['min_percentage']:.2f}%")
    print(f"  Average: {before_stats['avg_percentage']:.2f}%")
    print(f"  Overall: {before_stats['overall_percentage']:.2f}%")
    
    print(f"\nAfter Election ({after_stats['date_range']}):")
    print(f"  Max: {after_stats['max_percentage']:.2f}%")
    print(f"  Min: {after_stats['min_percentage']:.2f}%")
    print(f"  Average: {after_stats['avg_percentage']:.2f}%")
    print(f"  Overall: {after_stats['overall_percentage']:.2f}%")
    
    print(f"\nPercentage Change:")
    print(f"  Max: {comparison_stats['percentage_change']['max']:.2f}%")
    print(f"  Min: {comparison_stats['percentage_change']['min']:.2f}%")
    print(f"  Average: {comparison_stats['percentage_change']['avg']:.2f}%")
    print(f"  Overall: {comparison_stats['percentage_change']['overall']:.2f}%")
    
    return election_stats

if __name__ == "__main__":
    main()