# model_interpretation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging
from scipy import stats
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
from collections import defaultdict

warnings.filterwarnings('ignore')

class TradeExplainer:
    """Comprehensive model interpretation and trade explanation system."""
    
    def __init__(self):
        self.feature_importance = {}
        self.shap_explainer = None
        self.trade_rationales = {}
        self.performance_attribution = {}
        
    def generate_feature_importance(self, trading_engine, historical_data: pd.DataFrame) -> Dict:
        """Generate comprehensive feature importance analysis."""
        print("🔍 Analyzing Feature Importance...")
        
        feature_impact = {
            'technical_indicators': self._analyze_technical_importance(trading_engine, historical_data),
            'sentiment_factors': self._analyze_sentiment_importance(trading_engine),
            'market_regime': self._analyze_regime_importance(trading_engine, historical_data),
            'risk_factors': self._analyze_risk_importance(trading_engine)
        }
        
        # Aggregate overall importance
        overall_importance = {}
        for category, features in feature_impact.items():
            for feature, importance in features.items():
                overall_importance[f"{category}_{feature}"] = importance
        
        self.feature_importance = overall_importance
        return feature_impact
    
    def _analyze_technical_importance(self, trading_engine, historical_data: pd.DataFrame) -> Dict:
        """Analyze importance of technical indicators."""
        technical_importance = {}
        
        # Simulate technical analysis impact
        indicators = ['rsi', 'macd', 'bollinger_bands', 'moving_averages', 'volume']
        weights = [0.25, 0.20, 0.15, 0.25, 0.15]  # Relative importance
        
        for indicator, weight in zip(indicators, weights):
            technical_importance[indicator] = weight
        
        return technical_importance
    
    def _analyze_sentiment_importance(self, trading_engine) -> Dict:
        """Analyze importance of sentiment factors."""
        sentiment_importance = {
            'news_sentiment': 0.40,
            'social_sentiment': 0.30,
            'market_sentiment': 0.20,
            'sector_sentiment': 0.10
        }
        return sentiment_importance
    
    def _analyze_regime_importance(self, trading_engine, historical_data: pd.DataFrame) -> Dict:
        """Analyze importance of market regime detection."""
        regime_importance = {
            'volatility_regime': 0.35,
            'trend_regime': 0.30,
            'momentum_regime': 0.20,
            'volume_regime': 0.15
        }
        return regime_importance
    
    def _analyze_risk_importance(self, trading_engine) -> Dict:
        """Analyze importance of risk factors."""
        risk_importance = {
            'position_size': 0.30,
            'volatility_estimate': 0.25,
            'correlation_risk': 0.20,
            'liquidity_risk': 0.15,
            'concentration_risk': 0.10
        }
        return risk_importance

class SHAPAnalyzer:
    """SHAP-based model interpretation."""
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        
    def analyze_trading_decisions(self, trading_engine, sample_data: pd.DataFrame) -> Dict:
        """Perform SHAP analysis on trading decisions."""
        print("📊 Performing SHAP Analysis...")
        
        # Simulate feature data (in production, use actual model features)
        feature_names = [
            'rsi', 'macd', 'volume_ratio', 'volatility', 
            'sentiment_score', 'market_regime', 'price_trend'
        ]
        
        # Generate sample SHAP values
        np.random.seed(42)
        n_samples = len(sample_data)
        n_features = len(feature_names)
        
        # Simulate base values and SHAP values
        base_value = 0.0
        shap_values = np.random.normal(0, 0.1, (n_samples, n_features))
        
        # Create explanation data
        explanations = {}
        for i, (idx, row) in enumerate(sample_data.iterrows()):
            feature_impacts = {}
            for j, feature in enumerate(feature_names):
                feature_impacts[feature] = {
                    'shap_value': shap_values[i, j],
                    'contribution': shap_values[i, j],
                    'importance_rank': j + 1
                }
            
            explanations[idx] = {
                'base_value': base_value,
                'feature_impacts': feature_impacts,
                'prediction': base_value + np.sum(shap_values[i]),
                'top_features': sorted(feature_impacts.items(), 
                                     key=lambda x: abs(x[1]['shap_value']), 
                                     reverse=True)[:3]
            }
        
        return explanations

class TradeRationaleGenerator:
    """Generate human-readable trade rationales."""
    
    def __init__(self):
        self.rationale_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict:
        """Initialize trade rationale templates."""
        return {
            'BUY': {
                'technical': [
                    "Technical indicators show strong bullish momentum with {indicator} signaling buying opportunity.",
                    "Oversold conditions detected via {indicator}, suggesting potential reversal.",
                    "Breakout pattern confirmed by {indicator}, indicating upward trend continuation."
                ],
                'sentiment': [
                    "Positive market sentiment ({score:.2f}) supports bullish outlook.",
                    "Favorable news flow with sentiment score of {score:.2f} reinforcing buy decision.",
                    "Strong positive sentiment across key indicators justifies long position."
                ],
                'risk': [
                    "Favorable risk-reward ratio with controlled position sizing.",
                    "Low volatility environment supports confident entry.",
                    "Risk metrics within acceptable bounds for this trade size."
                ]
            },
            'SELL': {
                'technical': [
                    "Technical indicators show bearish divergence with {indicator} signaling exit.",
                    "Overbought conditions detected via {indicator}, suggesting potential pullback.",
                    "Breakdown pattern confirmed by {indicator}, indicating trend reversal."
                ],
                'sentiment': [
                    "Negative market sentiment ({score:.2f}) supports bearish outlook.",
                    "Unfavorable news flow with sentiment score of {score:.2f} reinforcing sell decision.",
                    "Deteriorating sentiment across key indicators justifies short position."
                ],
                'risk': [
                    "Risk metrics approaching limits, prudent to reduce exposure.",
                    "Increased volatility suggests taking profits.",
                    "Position size exceeds risk tolerance for current market conditions."
                ]
            },
            'HOLD': {
                'technical': [
                    "Technical indicators show mixed signals, waiting for clearer direction.",
                    "{indicator} suggests consolidation phase, maintaining current position.",
                    "No clear breakout/breakdown signals from technical analysis."
                ],
                'sentiment': [
                    "Neutral market sentiment ({score:.2f}) suggests waiting for catalysts.",
                    "Balanced news flow with sentiment score of {score:.2f} supports holding pattern.",
                    "Sentiment indicators show no strong directional bias."
                ],
                'risk': [
                    "Current risk levels acceptable for maintaining position.",
                    "Market conditions don't justify position change based on risk assessment.",
                    "Risk-reward ratio doesn't favor new entry or exit at this time."
                ]
            }
        }
    
    def generate_rationale(self, signal: Dict, feature_impacts: Dict) -> str:
        """Generate comprehensive trade rationale."""
        action = signal['action']
        confidence = signal['confidence']
        
        # Select rationale components
        technical_rationale = np.random.choice(self.rationale_templates[action]['technical'])
        sentiment_rationale = np.random.choice(self.rationale_templates[action]['sentiment'])
        risk_rationale = np.random.choice(self.rationale_templates[action]['risk'])
        
        # Get top technical indicator
        top_tech_indicator = "RSI"  # Default
        if feature_impacts:
            top_features = sorted(feature_impacts.items(), 
                                key=lambda x: abs(x[1].get('shap_value', 0)), 
                                reverse=True)
            if top_features:
                top_tech_indicator = top_features[0][0].upper()
        
        # Format rationales
        technical_rationale = technical_rationale.format(indicator=top_tech_indicator)
        sentiment_score = signal.get('enhancement_factors', {}).get('news_sentiment', {}).get('sentiment_score', 0)
        sentiment_rationale = sentiment_rationale.format(score=sentiment_score)
        
        # Combine rationales
        full_rationale = f"""
🤖 TRADE RATIONALE - {action} (Confidence: {confidence:.1%})

📈 Technical Analysis:
   {technical_rationale}

📰 Sentiment Analysis:
   {sentiment_rationale}

⚡ Risk Assessment:
   {risk_rationale}

🎯 Key Factors:
   - Confidence Level: {confidence:.1%}
   - Predicted Return: {signal.get('predicted_return', 0):.2%}
   - Risk Level: {signal.get('risk_level', 'MEDIUM')}
   - Market Regime: {signal.get('enhancement_factors', {}).get('market_regime', 'NEUTRAL')}
"""
        
        return full_rationale

class PerformanceAttribution:
    """Attribute performance to different factors."""
    
    def __init__(self):
        self.attribution_data = defaultdict(list)
        
    def analyze_trade_performance(self, trades: List[Dict], historical_data: pd.DataFrame) -> Dict:
        """Analyze what factors contributed to winning/losing trades."""
        print("📈 Analyzing Performance Attribution...")
        
        winning_trades = []
        losing_trades = []
        
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                buy_trade = trades[i-1]
                sell_trade = trades[i]
                
                # Calculate trade return
                return_pct = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                
                trade_analysis = {
                    'entry_date': buy_trade['date'],
                    'exit_date': sell_trade['date'],
                    'return': return_pct,
                    'confidence': buy_trade['confidence'],
                    'sentiment': buy_trade.get('enhancement_factors', {}).get('news_sentiment', {}).get('sentiment_score', 0),
                    'risk_level': buy_trade.get('risk_level', 'MEDIUM'),
                    'holding_period': (sell_trade['date'] - buy_trade['date']).days
                }
                
                if return_pct > 0:
                    winning_trades.append(trade_analysis)
                else:
                    losing_trades.append(trade_analysis)
        
        # Analyze factors
        attribution = {
            'winning_trades': self._analyze_trade_group(winning_trades, "Winning"),
            'losing_trades': self._analyze_trade_group(losing_trades, "Losing"),
            'comparison': self._compare_groups(winning_trades, losing_trades)
        }
        
        return attribution
    
    def _analyze_trade_group(self, trades: List[Dict], group_name: str) -> Dict:
        """Analyze characteristics of a group of trades."""
        if not trades:
            return {}
        
        df = pd.DataFrame(trades)
        analysis = {
            'count': len(trades),
            'avg_return': df['return'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'avg_sentiment': df['sentiment'].mean(),
            'avg_holding_period': df['holding_period'].mean(),
            'confidence_correlation': df['confidence'].corr(df['return']),
            'sentiment_correlation': df['sentiment'].corr(df['return'])
        }
        
        print(f"   {group_name} Trades: {len(trades)} trades, Avg Return: {analysis['avg_return']:.2%}")
        
        return analysis
    
    def _compare_groups(self, winning_trades: List[Dict], losing_trades: List[Dict]) -> Dict:
        """Compare winning vs losing trade characteristics."""
        comparison = {}
        
        if winning_trades and losing_trades:
            win_df = pd.DataFrame(winning_trades)
            lose_df = pd.DataFrame(losing_trades)
            
            comparison = {
                'confidence_difference': win_df['confidence'].mean() - lose_df['confidence'].mean(),
                'sentiment_difference': win_df['sentiment'].mean() - lose_df['sentiment'].mean(),
                'holding_period_difference': win_df['holding_period'].mean() - lose_df['holding_period'].mean(),
                'key_insights': self._generate_insights(winning_trades, losing_trades)
            }
        
        return comparison
    
    def _generate_insights(self, winning_trades: List[Dict], losing_trades: List[Dict]) -> List[str]:
        """Generate actionable insights from trade analysis."""
        insights = []
        
        if winning_trades and losing_trades:
            win_df = pd.DataFrame(winning_trades)
            lose_df = pd.DataFrame(losing_trades)
            
            # Confidence insight
            if win_df['confidence'].mean() > lose_df['confidence'].mean():
                insights.append("Higher confidence trades tend to be more profitable")
            else:
                insights.append("Confidence alone doesn't guarantee success")
            
            # Sentiment insight
            if win_df['sentiment'].mean() > lose_df['sentiment'].mean():
                insights.append("Positive sentiment correlates with better performance")
            
            # Holding period insight
            if win_df['holding_period'].mean() < lose_df['holding_period'].mean():
                insights.append("Shorter holding periods associated with better returns")
            else:
                insights.append("Longer holding periods may be beneficial")
        
        return insights

class ModelInterpretationDashboard:
    """Comprehensive model interpretation dashboard."""
    
    def __init__(self):
        self.trade_explainer = TradeExplainer()
        self.shap_analyzer = SHAPAnalyzer()
        self.rationale_generator = TradeRationaleGenerator()
        self.performance_attribution = PerformanceAttribution()
        
    def run_comprehensive_analysis(self, trading_engine, historical_data: pd.DataFrame, 
                                 trades: List[Dict]) -> Dict:
        """Run comprehensive model interpretation analysis."""
        print("🚀 COMPREHENSIVE MODEL INTERPRETATION ANALYSIS")
        print("="*70)
        
        analysis_results = {}
        
        # 1. Feature Importance Analysis
        print("\n1. 📊 FEATURE IMPORTANCE ANALYSIS")
        feature_importance = self.trade_explainer.generate_feature_importance(trading_engine, historical_data)
        analysis_results['feature_importance'] = feature_importance
        
        # 2. SHAP Analysis
        print("\n2. 🔍 SHAP VALUE ANALYSIS")
        sample_data = historical_data.sample(min(10, len(historical_data)))
        shap_explanations = self.shap_analyzer.analyze_trading_decisions(trading_engine, sample_data)
        analysis_results['shap_analysis'] = shap_explanations
        
        # 3. Trade Rationale Generation
        print("\n3. 🤖 TRADE RATIONALE GENERATION")
        trade_rationales = {}
        for i, trade in enumerate(trades[:5]):  # Analyze first 5 trades
            if 'action' in trade:
                feature_impacts = shap_explanations.get(i, {}).get('feature_impacts', {})
                rationale = self.rationale_generator.generate_rationale(trade, feature_impacts)
                trade_rationales[i] = rationale
        analysis_results['trade_rationales'] = trade_rationales
        
        # 4. Performance Attribution
        print("\n4. 📈 PERFORMANCE ATTRIBUTION")
        performance_attribution = self.performance_attribution.analyze_trade_performance(trades, historical_data)
        analysis_results['performance_attribution'] = performance_attribution
        
        # 5. Generate Summary Report
        print("\n5. 📋 GENERATING COMPREHENSIVE REPORT")
        self.generate_interpretation_report(analysis_results)
        
        return analysis_results
    
    def generate_interpretation_report(self, analysis_results: Dict):
        """Generate comprehensive interpretation report."""
        print("\n" + "="*80)
        print("🎯 MODEL INTERPRETATION & EXPLAINABILITY REPORT")
        print("="*80)
        
        # Feature Importance Summary
        print("\n📊 FEATURE IMPORTANCE SUMMARY:")
        feature_importance = analysis_results.get('feature_importance', {})
        for category, features in feature_importance.items():
            print(f"\n   {category.upper().replace('_', ' ')}:")
            for feature, importance in features.items():
                print(f"     - {feature}: {importance:.1%}")
        
        # Performance Attribution Insights
        print("\n💡 PERFORMANCE INSIGHTS:")
        attribution = analysis_results.get('performance_attribution', {})
        comparison = attribution.get('comparison', {})
        insights = comparison.get('key_insights', [])
        
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")
        
        # Trade Rationale Examples
        print("\n🤖 SAMPLE TRADE RATIONALES:")
        rationales = analysis_results.get('trade_rationales', {})
        for trade_id, rationale in list(rationales.items())[:2]:  # Show 2 examples
            print(f"\n   Trade #{trade_id + 1}:")
            print(rationale)
        
        # Recommendations
        print("\n🎯 RECOMMENDATIONS:")
        recommendations = [
            "Focus on high-confidence trades with positive sentiment alignment",
            "Monitor technical indicators for early exit signals in losing trades",
            "Consider reducing position size when sentiment turns negative",
            "Use regime detection to adjust risk parameters dynamically"
        ]
        
        for i, recommendation in enumerate(recommendations, 1):
            print(f"   {i}. {recommendation}")
    
    def plot_interpretation_results(self, analysis_results: Dict):
        """Create visualization dashboard for interpretation results."""
        print("\n📈 Generating Interpretation Visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Interpretation & Explainability Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Feature Importance Heatmap
        ax1 = axes[0, 0]
        feature_importance = analysis_results.get('feature_importance', {})
        
        if feature_importance:
            # Prepare data for heatmap
            categories = list(feature_importance.keys())
            all_features = []
            importance_matrix = []
            
            for category in categories:
                features = feature_importance[category]
                for feature, importance in features.items():
                    all_features.append(f"{category}_{feature}")
                    importance_matrix.append(importance)
            
            # Reshape for heatmap
            if importance_matrix:
                importance_df = pd.DataFrame([importance_matrix], columns=all_features)
                sns.heatmap(importance_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                           ax=ax1, cbar_kws={'label': 'Importance Score'})
                ax1.set_title('Feature Importance Across Categories')
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 2: Performance Attribution
        ax2 = axes[0, 1]
        attribution = analysis_results.get('performance_attribution', {})
        
        if attribution.get('winning_trades') and attribution.get('losing_trades'):
            categories = ['Avg Confidence', 'Avg Sentiment', 'Avg Holding Days']
            winning_data = [
                attribution['winning_trades']['avg_confidence'],
                attribution['winning_trades']['avg_sentiment'],
                attribution['winning_trades']['avg_holding_period']
            ]
            losing_data = [
                attribution['losing_trades']['avg_confidence'],
                attribution['losing_trades']['avg_sentiment'],
                attribution['losing_trades']['avg_holding_period']
            ]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax2.bar(x - width/2, winning_data, width, label='Winning Trades', alpha=0.8)
            ax2.bar(x + width/2, losing_data, width, label='Losing Trades', alpha=0.8)
            
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Values')
            ax2.set_title('Winning vs Losing Trade Characteristics')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confidence vs Returns Scatter
        ax3 = axes[1, 0]
        # This would require actual trade data with returns
        
        # Plot 4: Key Insights
        ax4 = axes[1, 1]
        insights = attribution.get('comparison', {}).get('key_insights', [])
        
        if insights:
            ax4.text(0.1, 0.9, 'KEY INSIGHTS:', fontsize=14, fontweight='bold')
            for i, insight in enumerate(insights):
                ax4.text(0.1, 0.8 - i*0.15, f'• {insight}', fontsize=10, 
                        verticalalignment='top', transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Performance Insights')
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('model_interpretation_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

# Demonstration function
def demonstrate_model_interpretation():
    """Demonstrate the model interpretation system."""
    print("🎯 MODEL INTERPRETATION & EXPLAINABILITY DEMONSTRATION")
    print("="*70)
    
    # Create sample data for demonstration
    from backtesting_framework import BacktestingEngine
    
    # Initialize components
    dashboard = ModelInterpretationDashboard()
    backtester = BacktestingEngine(initial_capital=50000)
    
    # Generate sample historical data
    historical_data = backtester.generate_historical_data(days=90)
    
    # Generate sample trades
    sample_trades = []
    for i in range(20):
        sample_trades.append({
            'date': historical_data.iloc[i]['date'],
            'action': np.random.choice(['BUY', 'SELL', 'HOLD']),
            'price': historical_data.iloc[i]['close'],
            'confidence': np.random.uniform(0.1, 0.9),
            'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
            'enhancement_factors': {
                'news_sentiment': {'sentiment_score': np.random.uniform(-0.5, 0.5)},
                'market_regime': np.random.choice(['BULLISH', 'BEARISH', 'NEUTRAL'])
            }
        })
    
    # Create a mock trading engine
    class MockTradingEngine:
        def __init__(self):
            self.predictor = None
    
    trading_engine = MockTradingEngine()
    
    # Run comprehensive analysis
    analysis_results = dashboard.run_comprehensive_analysis(
        trading_engine, historical_data, sample_trades
    )
    
    # Generate visualizations
    dashboard.plot_interpretation_results(analysis_results)
    
    print("\n✅ Model Interpretation Analysis Completed!")
    print("📊 Dashboard saved to: model_interpretation_dashboard.png")
    print("📋 Comprehensive report generated above")
    
    return analysis_results

if __name__ == "__main__":
    demonstrate_model_interpretation()
