#!/usr/bin/env python3
"""
Advanced batch processing system for handling multiple documents efficiently.
"""

import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
from queue import Queue, Empty
import time

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Advanced batch processing system for efficient document processing.
    """
    
    def __init__(self, max_workers: int = 4, batch_size: int = 10, memory_limit_mb: int = 400):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.processing_queue = Queue()
        self.results_queue = Queue()
        self.active_batches = {}
        self.batch_counter = 0
        self.processing_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'average_time': 0,
            'peak_memory': 0
        }
    
    def add_processing_job(self, job_data: Dict, priority: int = 5) -> str:
        """Add a job to the processing queue."""
        job_id = f"job_{int(time.time() * 1000)}_{self.batch_counter}"
        self.batch_counter += 1
        
        job = {
            'id': job_id,
            'data': job_data,
            'priority': priority,
            'created_at': datetime.now(),
            'status': 'queued'
        }
        
        self.processing_queue.put((priority, job))
        logger.debug(f"Added job to queue: {job_id}")
        return job_id
    
    def process_batch(self, processing_function: Callable, context: Dict = None) -> Dict:
        """Process a batch of jobs using the provided function."""
        batch_id = f"batch_{self.batch_counter}"
        self.batch_counter += 1
        
        batch_info = {
            'id': batch_id,
            'start_time': datetime.now(),
            'jobs': [],
            'results': [],
            'errors': [],
            'status': 'processing'
        }
        
        self.active_batches[batch_id] = batch_info
        
        try:
            # Collect jobs for this batch
            jobs_to_process = self._collect_batch_jobs()
            
            if not jobs_to_process:
                batch_info['status'] = 'empty'
                return batch_info
            
            batch_info['jobs'] = [job['id'] for job in jobs_to_process]
            logger.info(f"Processing batch {batch_id} with {len(jobs_to_process)} jobs")
            
            # Process jobs in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(self._safe_process_job, job, processing_function, context): job
                    for job in jobs_to_process
                }
                
                # Collect results
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result = future.result()
                        if result['success']:
                            batch_info['results'].append(result)
                            self.processing_stats['successful'] += 1
                        else:
                            batch_info['errors'].append(result)
                            self.processing_stats['failed'] += 1
                            
                        self.processing_stats['total_processed'] += 1
                        
                    except Exception as e:
                        error_result = {
                            'job_id': job['id'],
                            'success': False,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        }
                        batch_info['errors'].append(error_result)
                        self.processing_stats['failed'] += 1
                        logger.error(f"Batch processing error for job {job['id']}: {e}")
            
            # Finalize batch
            batch_info['end_time'] = datetime.now()
            batch_info['duration'] = (batch_info['end_time'] - batch_info['start_time']).total_seconds()
            batch_info['status'] = 'completed'
            batch_info['success_rate'] = len(batch_info['results']) / len(jobs_to_process) * 100
            
            # Update average time
            if self.processing_stats['total_processed'] > 0:
                total_time = sum(b.get('duration', 0) for b in self.active_batches.values())
                self.processing_stats['average_time'] = total_time / len(self.active_batches)
            
            logger.info(f"Batch {batch_id} completed: {len(batch_info['results'])} successful, {len(batch_info['errors'])} errors")
            
        except Exception as e:
            batch_info['status'] = 'error'
            batch_info['error'] = str(e)
            logger.error(f"Batch {batch_id} failed: {e}")
        
        return batch_info
    
    def _collect_batch_jobs(self) -> List[Dict]:
        """Collect jobs for a batch."""
        jobs = []
        
        # Collect up to batch_size jobs, prioritizing by priority
        collected_jobs = []
        while len(collected_jobs) < self.batch_size:
            try:
                priority, job = self.processing_queue.get_nowait()
                collected_jobs.append((priority, job))
            except Empty:
                break
        
        # Sort by priority (lower number = higher priority)
        collected_jobs.sort(key=lambda x: x[0])
        jobs = [job for priority, job in collected_jobs]
        
        return jobs
    
    def _safe_process_job(self, job: Dict, processing_function: Callable, context: Dict) -> Dict:
        """Safely process a single job with error handling."""
        job_id = job['id']
        start_time = time.time()
        
        try:
            logger.debug(f"Processing job: {job_id}")
            
            # Add job context
            job_context = {
                'job_id': job_id,
                'batch_context': context or {},
                'start_time': start_time
            }
            
            # Process the job
            result = processing_function(job['data'], job_context)
            
            # Wrap result
            wrapped_result = {
                'job_id': job_id,
                'success': True,
                'result': result,
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Job {job_id} completed successfully in {wrapped_result['processing_time']:.2f}s")
            return wrapped_result
            
        except Exception as e:
            error_result = {
                'job_id': job_id,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.error(f"Job {job_id} failed: {e}")
            return error_result
    
    def process_documents_batch(self, documents: List[str], personas: List[str], 
                              processing_function: Callable) -> Dict:
        """Process multiple documents with multiple personas in batches."""
        batch_id = f"doc_batch_{self.batch_counter}"
        self.batch_counter += 1
        
        logger.info(f"Starting document batch processing: {len(documents)} docs, {len(personas)} personas")
        
        # Create all job combinations
        all_jobs = []
        for doc_path in documents:
            for persona in personas:
                job_data = {
                    'document_path': doc_path,
                    'persona': persona,
                    'type': 'document_processing'
                }
                job_id = self.add_processing_job(job_data, priority=1)
                all_jobs.append(job_id)
        
        # Process in batches
        batch_results = []
        total_jobs = len(all_jobs)
        processed_jobs = 0
        
        while processed_jobs < total_jobs:
            batch_info = self.process_batch(processing_function, {
                'batch_type': 'document_processing',
                'total_jobs': total_jobs,
                'processed_jobs': processed_jobs
            })
            
            batch_results.append(batch_info)
            processed_jobs += len(batch_info.get('jobs', []))
            
            logger.info(f"Document batch progress: {processed_jobs}/{total_jobs} jobs completed")
        
        # Compile final results
        final_results = {
            'batch_id': batch_id,
            'total_documents': len(documents),
            'total_personas': len(personas),
            'total_jobs': total_jobs,
            'batches': batch_results,
            'summary': self._compile_batch_summary(batch_results),
            'completed_at': datetime.now().isoformat()
        }
        
        return final_results
    
    def _compile_batch_summary(self, batch_results: List[Dict]) -> Dict:
        """Compile summary statistics from batch results."""
        total_successful = sum(len(batch.get('results', [])) for batch in batch_results)
        total_failed = sum(len(batch.get('errors', [])) for batch in batch_results)
        total_jobs = total_successful + total_failed
        
        total_time = sum(batch.get('duration', 0) for batch in batch_results)
        
        success_rates = [batch.get('success_rate', 0) for batch in batch_results if batch.get('success_rate') is not None]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0
        
        return {
            'total_jobs': total_jobs,
            'successful_jobs': total_successful,
            'failed_jobs': total_failed,
            'overall_success_rate': (total_successful / total_jobs * 100) if total_jobs > 0 else 0,
            'average_batch_success_rate': avg_success_rate,
            'total_processing_time': total_time,
            'average_job_time': total_time / total_jobs if total_jobs > 0 else 0,
            'throughput_jobs_per_second': total_jobs / total_time if total_time > 0 else 0
        }
    
    def process_persona_variations(self, base_persona: str, variations: List[Dict], 
                                 documents: List[str], processing_function: Callable) -> Dict:
        """Process multiple persona variations across documents."""
        logger.info(f"Processing persona variations: base='{base_persona[:50]}...', {len(variations)} variations, {len(documents)} docs")
        
        # Create persona-document combinations
        all_jobs = []
        for variation in variations:
            modified_persona = self._apply_persona_variation(base_persona, variation)
            for doc_path in documents:
                job_data = {
                    'document_path': doc_path,
                    'persona': modified_persona,
                    'variation_info': variation,
                    'base_persona': base_persona,
                    'type': 'persona_variation'
                }
                job_id = self.add_processing_job(job_data, priority=2)
                all_jobs.append(job_id)
        
        # Process all jobs
        results = []
        while not self.processing_queue.empty():
            batch_info = self.process_batch(processing_function, {
                'batch_type': 'persona_variation',
                'base_persona': base_persona[:50] + "..."
            })
            results.append(batch_info)
        
        return {
            'base_persona': base_persona,
            'variations_tested': len(variations),
            'documents_processed': len(documents),
            'batch_results': results,
            'summary': self._compile_batch_summary(results),
            'completed_at': datetime.now().isoformat()
        }
    
    def _apply_persona_variation(self, base_persona: str, variation: Dict) -> str:
        """Apply a variation to a base persona."""
        modified_persona = base_persona
        
        # Apply different types of variations
        if variation.get('type') == 'role_change':
            # Replace job role
            old_role = variation.get('old_role', '')
            new_role = variation.get('new_role', '')
            if old_role and new_role:
                modified_persona = modified_persona.replace(old_role, new_role)
        
        elif variation.get('type') == 'experience_level':
            # Modify experience level
            experience_terms = {
                'junior': ['entry-level', 'beginner', 'new'],
                'senior': ['experienced', 'expert', 'seasoned'],
                'lead': ['leadership', 'management', 'director']
            }
            
            target_level = variation.get('level', 'senior')
            if target_level in experience_terms:
                # Add experience descriptors
                terms = experience_terms[target_level]
                modified_persona += f" I am an {terms[0]} professional with {target_level}-level expertise."
        
        elif variation.get('type') == 'interest_focus':
            # Add specific interest focus
            focus_area = variation.get('focus', '')
            if focus_area:
                modified_persona += f" I am particularly interested in {focus_area} aspects."
        
        elif variation.get('type') == 'custom_addition':
            # Add custom text
            addition = variation.get('text', '')
            if addition:
                modified_persona += f" {addition}"
        
        return modified_persona
    
    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics."""
        return {
            'processing_stats': self.processing_stats.copy(),
            'active_batches': len(self.active_batches),
            'queue_size': self.processing_queue.qsize(),
            'configuration': {
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'memory_limit_mb': self.memory_limit_bytes / (1024 * 1024)
            },
            'recent_batches': [
                {
                    'id': batch_id,
                    'status': batch_info.get('status', 'unknown'),
                    'jobs_count': len(batch_info.get('jobs', [])),
                    'success_rate': batch_info.get('success_rate', 0),
                    'duration': batch_info.get('duration', 0)
                }
                for batch_id, batch_info in list(self.active_batches.items())[-5:]
            ]
        }
    
    def clear_completed_batches(self, keep_recent: int = 10):
        """Clear old completed batches to free memory."""
        completed_batches = [
            (batch_id, batch_info) 
            for batch_id, batch_info in self.active_batches.items()
            if batch_info.get('status') in ['completed', 'error', 'empty']
        ]
        
        # Sort by completion time and keep only recent ones
        completed_batches.sort(
            key=lambda x: x[1].get('end_time', x[1].get('start_time', datetime.min)),
            reverse=True
        )
        
        # Remove old batches
        for batch_id, _ in completed_batches[keep_recent:]:
            del self.active_batches[batch_id]
        
        logger.info(f"Cleared {len(completed_batches) - keep_recent} old batches")

# Global batch processor instance
batch_processor = BatchProcessor()
