#!/usr/bin/env python3
"""
Performance monitoring and optimization recommendations system.
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Container for a single performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str
    details: Dict[str, Any] = None

class PerformanceMonitor:
    """
    Advanced performance monitoring system for the document processing pipeline.
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.stage_timings: Dict[str, List[float]] = {}
        self.memory_usage: List[Dict[str, float]] = []
        self.optimization_history: List[Dict] = []
        self.monitoring_active = False
        self.start_time = None
        self.process = psutil.Process()
        
        # Performance thresholds
        self.thresholds = {
            'memory_mb': 500,  # MB
            'cpu_percent': 80,  # %
            'stage_time_seconds': 30,  # seconds
            'total_time_seconds': 300,  # 5 minutes
            'cache_hit_rate': 70  # %
        }
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_active = True
        self.start_time = datetime.now()
        logger.info("Performance monitoring started")
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.start_time:
            total_time = (datetime.now() - self.start_time).total_seconds()
            self.record_metric("total_pipeline_time", total_time, "seconds", "overall")
        logger.info("Performance monitoring stopped")
    
    def record_metric(self, name: str, value: float, unit: str, category: str, details: Dict = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            details=details or {}
        )
        self.metrics.append(metric)
        
        # Also store in stage timings for analysis
        if category == "timing" and name.endswith("_time"):
            stage_name = name.replace("_time", "")
            if stage_name not in self.stage_timings:
                self.stage_timings[stage_name] = []
            self.stage_timings[stage_name].append(value)
    
    def start_stage_timer(self, stage_name: str) -> Dict:
        """Start timing a pipeline stage."""
        return {
            'stage': stage_name,
            'start_time': time.time(),
            'start_memory': self.process.memory_info().rss / 1024 / 1024  # MB
        }
    
    def end_stage_timer(self, timer_data: Dict) -> float:
        """End timing a pipeline stage and record metrics."""
        end_time = time.time()
        duration = end_time - timer_data['start_time']
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - timer_data['start_memory']
        
        stage_name = timer_data['stage']
        
        # Record timing metric
        self.record_metric(
            f"{stage_name}_time",
            duration,
            "seconds",
            "timing",
            {"memory_delta_mb": memory_delta}
        )
        
        # Record memory metric
        self.record_metric(
            f"{stage_name}_memory_delta",
            memory_delta,
            "MB",
            "memory"
        )
        
        logger.debug(f"Stage '{stage_name}' completed in {duration:.2f}s, memory delta: {memory_delta:.1f}MB")
        return duration
    
    def _background_monitor(self):
        """Background thread for continuous monitoring."""
        while self.monitoring_active:
            try:
                # Record system metrics
                cpu_percent = self.process.cpu_percent()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                
                self.record_metric("cpu_usage", cpu_percent, "percent", "system")
                self.record_metric("memory_usage", memory_mb, "MB", "system")
                
                # Store memory usage history
                self.memory_usage.append({
                    'timestamp': datetime.now().isoformat(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent
                })
                
                # Check for performance issues
                self._check_performance_thresholds(cpu_percent, memory_mb)
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.warning(f"Background monitoring error: {e}")
                time.sleep(10)
    
    def _check_performance_thresholds(self, cpu_percent: float, memory_mb: float):
        """Check if performance metrics exceed thresholds."""
        issues = []
        
        if memory_mb > self.thresholds['memory_mb']:
            issues.append({
                'type': 'high_memory',
                'value': memory_mb,
                'threshold': self.thresholds['memory_mb'],
                'severity': 'warning' if memory_mb < self.thresholds['memory_mb'] * 1.5 else 'critical'
            })
        
        if cpu_percent > self.thresholds['cpu_percent']:
            issues.append({
                'type': 'high_cpu',
                'value': cpu_percent,
                'threshold': self.thresholds['cpu_percent'],
                'severity': 'warning'
            })
        
        if issues:
            self.record_metric(
                "performance_issues",
                len(issues),
                "count",
                "alerts",
                {"issues": issues}
            )
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if not self.metrics:
            return {"status": "no_data", "message": "No performance data available"}
        
        # Calculate summary statistics
        summary = {
            "monitoring_duration": self._get_monitoring_duration(),
            "stage_performance": self._analyze_stage_performance(),
            "resource_usage": self._analyze_resource_usage(),
            "optimization_recommendations": self._generate_recommendations(),
            "performance_score": self._calculate_performance_score()
        }
        
        return summary
    
    def _get_monitoring_duration(self) -> Dict:
        """Get total monitoring duration."""
        if not self.start_time:
            return {"status": "not_started"}
        
        if self.monitoring_active:
            duration = (datetime.now() - self.start_time).total_seconds()
        else:
            # Find last metric timestamp
            last_metric = max(self.metrics, key=lambda m: m.timestamp)
            duration = (last_metric.timestamp - self.start_time).total_seconds()
        
        return {
            "total_seconds": duration,
            "formatted": f"{duration:.1f}s",
            "status": "active" if self.monitoring_active else "completed"
        }
    
    def _analyze_stage_performance(self) -> Dict:
        """Analyze performance of each pipeline stage."""
        stage_analysis = {}
        
        for stage, timings in self.stage_timings.items():
            if timings:
                stage_analysis[stage] = {
                    "avg_time": np.mean(timings),
                    "min_time": np.min(timings),
                    "max_time": np.max(timings),
                    "std_time": np.std(timings),
                    "executions": len(timings),
                    "total_time": np.sum(timings),
                    "performance_rating": self._rate_stage_performance(stage, np.mean(timings))
                }
        
        return stage_analysis
    
    def _analyze_resource_usage(self) -> Dict:
        """Analyze system resource usage."""
        if not self.memory_usage:
            return {"status": "no_data"}
        
        memory_values = [entry['memory_mb'] for entry in self.memory_usage]
        cpu_values = [entry['cpu_percent'] for entry in self.memory_usage]
        
        return {
            "memory": {
                "avg_mb": np.mean(memory_values),
                "max_mb": np.max(memory_values),
                "min_mb": np.min(memory_values),
                "peak_usage_rating": self._rate_memory_usage(np.max(memory_values))
            },
            "cpu": {
                "avg_percent": np.mean(cpu_values),
                "max_percent": np.max(cpu_values),
                "sustained_high_cpu": len([x for x in cpu_values if x > 80]) / len(cpu_values) * 100
            }
        }
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        # Analyze stage timings
        for stage, timings in self.stage_timings.items():
            if timings:
                avg_time = np.mean(timings)
                if avg_time > self.thresholds['stage_time_seconds']:
                    recommendations.append({
                        "type": "slow_stage",
                        "stage": stage,
                        "issue": f"Stage '{stage}' averaging {avg_time:.1f}s (threshold: {self.thresholds['stage_time_seconds']}s)",
                        "recommendations": self._get_stage_optimization_tips(stage),
                        "priority": "high" if avg_time > self.thresholds['stage_time_seconds'] * 2 else "medium"
                    })
        
        # Analyze memory usage
        if self.memory_usage:
            max_memory = max(entry['memory_mb'] for entry in self.memory_usage)
            if max_memory > self.thresholds['memory_mb']:
                recommendations.append({
                    "type": "high_memory",
                    "issue": f"Peak memory usage: {max_memory:.1f}MB (threshold: {self.thresholds['memory_mb']}MB)",
                    "recommendations": [
                        "Implement batch processing for large documents",
                        "Enable smart caching with size limits",
                        "Process documents sequentially instead of parallel",
                        "Clear intermediate results more frequently"
                    ],
                    "priority": "high" if max_memory > self.thresholds['memory_mb'] * 1.5 else "medium"
                })
        
        # Check for cache performance
        cache_metrics = [m for m in self.metrics if m.name == "cache_hit_rate"]
        if cache_metrics:
            latest_hit_rate = cache_metrics[-1].value
            if latest_hit_rate < self.thresholds['cache_hit_rate']:
                recommendations.append({
                    "type": "low_cache_efficiency",
                    "issue": f"Cache hit rate: {latest_hit_rate:.1f}% (threshold: {self.thresholds['cache_hit_rate']}%)",
                    "recommendations": [
                        "Increase cache size limits",
                        "Optimize cache key generation",
                        "Review cache expiration policies",
                        "Pre-warm cache with common queries"
                    ],
                    "priority": "medium"
                })
        
        return recommendations
    
    def _get_stage_optimization_tips(self, stage: str) -> List[str]:
        """Get optimization tips for specific stages."""
        tips_map = {
            "document_reading": [
                "Implement parallel PDF processing",
                "Cache parsed document structures",
                "Optimize text extraction algorithms",
                "Use streaming for large documents"
            ],
            "embedding_generation": [
                "Batch embedding requests",
                "Cache embeddings with smart keys",
                "Use GPU acceleration if available",
                "Implement incremental embedding updates"
            ],
            "section_selection": [
                "Optimize similarity calculations",
                "Implement early termination for low scores",
                "Cache persona embeddings",
                "Use approximate nearest neighbor search"
            ],
            "subsection_refinement": [
                "Parallelize refinement operations",
                "Cache refinement results",
                "Optimize text cleaning algorithms",
                "Implement progressive refinement"
            ]
        }
        
        return tips_map.get(stage, [
            "Profile specific operations within this stage",
            "Implement caching for repeated operations",
            "Consider algorithmic optimizations",
            "Add progress tracking for long operations"
        ])
    
    def _rate_stage_performance(self, stage: str, avg_time: float) -> str:
        """Rate the performance of a stage."""
        if avg_time < 5:
            return "excellent"
        elif avg_time < 15:
            return "good"
        elif avg_time < 30:
            return "acceptable"
        else:
            return "needs_improvement"
    
    def _rate_memory_usage(self, max_memory: float) -> str:
        """Rate memory usage efficiency."""
        if max_memory < 200:
            return "excellent"
        elif max_memory < 400:
            return "good"
        elif max_memory < 600:
            return "acceptable"
        else:
            return "high"
    
    def _calculate_performance_score(self) -> Dict:
        """Calculate overall performance score."""
        score_components = {}
        total_score = 0
        weight_sum = 0
        
        # Stage timing score (40% weight)
        if self.stage_timings:
            stage_scores = []
            for stage, timings in self.stage_timings.items():
                avg_time = np.mean(timings)
                # Score inversely proportional to time (normalized)
                stage_score = max(0, 100 - (avg_time / self.thresholds['stage_time_seconds'] * 100))
                stage_scores.append(stage_score)
            
            timing_score = np.mean(stage_scores)
            score_components['timing'] = timing_score
            total_score += timing_score * 0.4
            weight_sum += 0.4
        
        # Memory efficiency score (30% weight)
        if self.memory_usage:
            max_memory = max(entry['memory_mb'] for entry in self.memory_usage)
            memory_score = max(0, 100 - (max_memory / self.thresholds['memory_mb'] * 100))
            score_components['memory'] = memory_score
            total_score += memory_score * 0.3
            weight_sum += 0.3
        
        # Cache efficiency score (20% weight)
        cache_metrics = [m for m in self.metrics if m.name == "cache_hit_rate"]
        if cache_metrics:
            cache_score = cache_metrics[-1].value
            score_components['cache'] = cache_score
            total_score += cache_score * 0.2
            weight_sum += 0.2
        
        # Stability score (10% weight)
        error_metrics = [m for m in self.metrics if m.category == "alerts"]
        if error_metrics:
            stability_score = max(0, 100 - len(error_metrics) * 10)  # -10 per issue
        else:
            stability_score = 100
        
        score_components['stability'] = stability_score
        total_score += stability_score * 0.1
        weight_sum += 0.1
        
        # Normalize score
        if weight_sum > 0:
            final_score = total_score / weight_sum
        else:
            final_score = 0
        
        return {
            "overall_score": round(final_score, 1),
            "components": score_components,
            "rating": self._get_score_rating(final_score),
            "explanation": self._get_score_explanation(final_score, score_components)
        }
    
    def _get_score_rating(self, score: float) -> str:
        """Get human-readable rating for performance score."""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "acceptable"
        elif score >= 40:
            return "needs_improvement"
        else:
            return "poor"
    
    def _get_score_explanation(self, score: float, components: Dict) -> str:
        """Generate explanation for the performance score."""
        explanations = []
        
        for component, value in components.items():
            if value < 60:
                explanations.append(f"Low {component} performance ({value:.1f}/100)")
            elif value > 85:
                explanations.append(f"Excellent {component} performance ({value:.1f}/100)")
        
        if not explanations:
            return f"Overall performance score of {score:.1f}/100 indicates balanced system performance."
        
        return f"Score of {score:.1f}/100. Issues: {'; '.join(explanations)}"
    
    def export_performance_data(self, filepath: str):
        """Export performance data to JSON file."""
        try:
            export_data = {
                "summary": self.get_performance_summary(),
                "detailed_metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "category": m.category,
                        "timestamp": m.timestamp.isoformat(),
                        "details": m.details
                    }
                    for m in self.metrics
                ],
                "memory_timeline": self.memory_usage,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Performance data exported to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export performance data: {e}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
