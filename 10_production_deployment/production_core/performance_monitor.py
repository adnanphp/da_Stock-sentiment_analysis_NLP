# performance_monitor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """
    Real-time performance monitoring and alerting system.
    """
    
    def __init__(self, trading_engine):
        self.engine = trading_engine
        self.performance_history = []
        self.alerts = []
        
    def generate_real_time_dashboard(self):
        """Generate real-time performance dashboard."""
        portfolio_summary = self.engine.get_portfolio_summary()
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('LIVE TRADING PERFORMANCE DASHBOARD', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        if self.engine.performance_log:
            timestamps = [log['timestamp'] for log in self.engine.performance_log]
            values = [log['portfolio_value'] for log in self.engine.performance_log]
            
            axes[0, 0].plot(timestamps, values, linewidth=2, color='blue')
            axes[0, 0].set_title('Portfolio Value Over Time', fontweight='bold')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add initial capital line
            axes[0, 0].axhline(y=self.engine.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
            axes[0, 0].legend()
        
        # 2. Daily P&L
        daily_pnl = self._calculate_daily_pnl()
        if daily_pnl:
            dates = list(daily_pnl.keys())
            pnls = list(daily_pnl.values())
            
            colors = ['green' if pnl > 0 else 'red' for pnl in pnls]
            axes[0, 1].bar(dates, pnls, color=colors, alpha=0.7)
            axes[0, 1].set_title('Daily P&L', fontweight='bold')
            axes[0, 1].set_ylabel('P&L ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Trade Distribution
        if self.engine.trade_history:
            actions = [trade['action'] for trade in self.engine.trade_history]
            action_counts = pd.Series(actions).value_counts()
            
            axes[0, 2].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 2].set_title('Trade Distribution', fontweight='bold')
        
        # 4. Risk Metrics
        risk_metrics = self._calculate_risk_metrics()
        metrics_data = [
            ['Total Return', f"{portfolio_summary['total_return']:.2%}"],
            ['Sharpe Ratio', f"{portfolio_summary['sharpe_ratio']:.2f}"],
            ['Max Drawdown', f"{risk_metrics.get('max_drawdown', 0):.2%}"],
            ['Win Rate', f"{portfolio_summary['daily_win_rate']:.1%}"],
            ['Volatility', f"{risk_metrics.get('volatility', 0):.2%}"],
            ['Total Trades', f"{portfolio_summary['total_trades']}"]
        ]
        
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=metrics_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 0].set_title('Performance Metrics', fontweight='bold')
        
        # 5. Position Sizing
        if self.engine.trade_history:
            position_sizes = [trade['quantity'] * trade['price'] for trade in self.engine.trade_history]
            axes[1, 1].hist(position_sizes, bins=20, alpha=0.7, color='orange')
            axes[1, 1].set_title('Position Size Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Position Size ($)')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. Alerts
        recent_alerts = self.alerts[-10:]  # Last 10 alerts
        if recent_alerts:
            alert_text = "\n".join([f"• {alert}" for alert in recent_alerts])
            axes[1, 2].text(0.1, 0.9, alert_text, transform=axes[1, 2].transAxes, fontsize=9, verticalalignment='top')
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Recent Alerts', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('live_performance_dashboard.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("📊 Live performance dashboard saved: 'live_performance_dashboard.png'")
    
    def _calculate_daily_pnl(self):
        """Calculate daily P&L from trade history."""
        daily_pnl = {}
        for trade in self.engine.trade_history:
            date = trade['timestamp'].date()
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += trade['pnl']
        return daily_pnl
    
    def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics."""
        if not self.engine.performance_log:
            return {}
        
        portfolio_values = [log['portfolio_value'] for log in self.engine.performance_log]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 and np.std(returns) > 0 else 0
        
        # Calculate drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def check_alerts(self):
        """Check for performance alerts."""
        portfolio_summary = self.engine.get_portfolio_summary()
        risk_breaches, risk_metrics = self.engine.check_risk_limits()
        
        alerts = []
        
        # Performance alerts
        if portfolio_summary['total_return'] < -0.05:  # 5% loss
            alerts.append(f"Portfolio down {portfolio_summary['total_return']:.2%}")
        
        if portfolio_summary['daily_win_rate'] < 0.3:  # 30% win rate
            alerts.append(f"Low win rate: {portfolio_summary['daily_win_rate']:.1%}")
        
        # Risk alerts
        alerts.extend(risk_breaches)
        
        # Model performance alerts
        if len(self.engine.performance_log) > 10:
            recent_errors = [log.get('prediction_error', 0) for log in self.engine.performance_log[-10:]]
            avg_error = np.mean(recent_errors)
            if avg_error > 0.1:  # 10% average error
                alerts.append(f"High prediction error: {avg_error:.2%}")
        
        # Log new alerts
        for alert in alerts:
            if alert not in self.alerts:
                self.alerts.append(f"{datetime.now().strftime('%H:%M:%S')} - {alert}")
                print(f"🚨 ALERT: {alert}")
        
        return alerts
    
    def generate_daily_report(self):
        """Generate daily performance report."""
        portfolio_summary = self.engine.get_portfolio_summary()
        risk_metrics = self._calculate_risk_metrics()
        alerts = self.check_alerts()
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_value': portfolio_summary['current_value'],
            'daily_return': portfolio_summary['total_return'],
            'cash_balance': portfolio_summary['cash'],
            'active_positions': len(portfolio_summary['active_positions']),
            'total_trades': portfolio_summary['total_trades'],
            'win_rate': portfolio_summary['daily_win_rate'],
            'sharpe_ratio': portfolio_summary['sharpe_ratio'],
            'max_drawdown': risk_metrics.get('max_drawdown', 0),
            'volatility': risk_metrics.get('volatility', 0),
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        with open(f'daily_report_{datetime.now().strftime("%Y%m%d")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Daily report saved: daily_report_{datetime.now().strftime('%Y%m%d')}.json")
        return report

# Real-time Monitoring System
class RealTimeMonitor:
    def __init__(self, trading_manager, check_interval=60):
        self.manager = trading_manager
        self.monitor = PerformanceMonitor(trading_manager.engine)
        self.check_interval = check_interval
        self.is_monitoring = False
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        self.is_monitoring = True
        print("🔍 STARTING REAL-TIME MONITORING")
        
        check_count = 0
        while self.is_monitoring:
            try:
                check_count += 1
                print(f"\n📊 Monitoring Check {check_count} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Check for alerts
                alerts = self.monitor.check_alerts()
                
                # Generate dashboard every 5 checks
                if check_count % 5 == 0:
                    self.monitor.generate_real_time_dashboard()
                
                # Generate daily report at market close (4 PM)
                if datetime.now().hour == 16 and datetime.now().minute < 5:
                    self.monitor.generate_daily_report()
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        print("🛑 Real-time monitoring stopped")

if __name__ == "__main__":
    # Demo the monitoring system
    from live_trading_engine import LiveTradingManager
    
    # Initialize with small capital for demo
    manager = LiveTradingManager(initial_capital=1000)
    monitor = RealTimeMonitor(manager, check_interval=30)
    
    print("🧪 DEMO: REAL-TIME MONITORING SYSTEM")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        # Run monitoring for 2 minutes for demo
        import threading
        
        def stop_after_delay():
            time.sleep(120)  # 2 minutes
            monitor.stop_monitoring()
        
        stop_thread = threading.Thread(target=stop_after_delay)
        stop_thread.daemon = True
        stop_thread.start()
        
        monitor.start_monitoring()
        
    except KeyboardInterrupt:
        monitor.stop_monitoring()
