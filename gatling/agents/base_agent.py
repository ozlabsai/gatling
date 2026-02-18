"""
Base Agent Class for Project Gatling Multi-Agent System

This module provides the foundational GatlingAgent class that all
specialized agents (LatentSubstrateAgent, EnergyGeometryAgent, etc.) inherit from.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class GatlingAgent(ABC):
    """
    Base class for all Gatling agents.
    
    Each specialized agent (LatentSubstrateAgent, EnergyGeometryAgent, etc.)
    inherits from this class and implements the execute_task method.
    
    Attributes:
        name: Unique agent identifier (e.g., "latent_substrate_agent")
        workstream: Which research workstream this agent belongs to
        task_queue_path: Path to the global task queue
        output_dir: Where this agent writes its outputs
        logger: Logging instance for this agent
    """
    
    def __init__(self, agent_name: str, workstream: str):
        """
        Initialize a Gatling agent.
        
        Args:
            agent_name: Unique identifier for this agent
            workstream: Research workstream (latent_substrate, energy_geometry, etc.)
        """
        self.name = agent_name
        self.workstream = workstream
        self.task_queue_path = Path("tasks/task_queue.json")
        self.output_dir = Path(f"outputs/{workstream}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Initialized {self.name} for workstream '{self.workstream}'")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        # Create logs directory if needed
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        fh = logging.FileHandler(log_dir / f"{self.name}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    @abstractmethod
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main task execution logic - must be implemented by each agent.
        
        Args:
            task: Task specification dictionary containing:
                - id: Unique task identifier
                - title: Human-readable task description
                - dependencies: List of task IDs that must complete first
                - acceptance_criteria: Success conditions
                - estimated_tokens: Expected computational budget
        
        Returns:
            Artifact dictionary containing:
                - component: Name of the implemented component
                - version: Semantic version
                - outputs: Paths to code, tests, docs
                - interface: API specification
                - validation_status: "passed" or "failed"
        """
        pass
    
    def load_task(self, task_id: str) -> Dict[str, Any]:
        """
        Load a specific task from the task queue.
        
        Args:
            task_id: Task identifier (e.g., "LSA-001")
        
        Returns:
            Task specification dictionary
        
        Raises:
            ValueError: If task_id not found in queue
        """
        if not self.task_queue_path.exists():
            raise FileNotFoundError(f"Task queue not found at {self.task_queue_path}")
        
        with open(self.task_queue_path) as f:
            queue = json.load(f)
        
        # Search all task lists
        for status in ["pending", "ready", "in_progress", "review", "completed"]:
            tasks = queue.get(status, [])
            for task in tasks:
                if task["id"] == task_id:
                    self.logger.info(f"Loaded task {task_id} (status: {status})")
                    return task
        
        raise ValueError(f"Task {task_id} not found in queue")
    
    def update_task_status(self, task_id: str, new_status: TaskStatus, 
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Update task status in the queue.
        
        Args:
            task_id: Task to update
            new_status: New status to set
            metadata: Optional additional metadata (e.g., error messages, progress)
        """
        with open(self.task_queue_path, 'r') as f:
            queue = json.load(f)
        
        # Find and move task
        task = None
        old_status = None
        
        for status in ["pending", "ready", "in_progress", "review", "completed", "failed"]:
            tasks = queue.get(status, [])
            for i, t in enumerate(tasks):
                if t["id"] == task_id:
                    task = tasks.pop(i)
                    old_status = status
                    break
            if task:
                break
        
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        # Update metadata
        if metadata:
            task.update(metadata)
        
        task["last_updated"] = datetime.now().isoformat()
        task["updated_by"] = self.name
        
        # Add to new status list
        new_status_key = new_status.value
        if new_status_key not in queue:
            queue[new_status_key] = []
        queue[new_status_key].append(task)
        
        # Write back
        with open(self.task_queue_path, 'w') as f:
            json.dump(queue, f, indent=2)
        
        self.logger.info(f"Updated task {task_id}: {old_status} -> {new_status_key}")
    
    def create_artifact(self, task_id: str, component_name: str, 
                       outputs: Dict[str, str], interface: Dict[str, Any],
                       version: str = "0.1.0") -> Dict[str, Any]:
        """
        Create an artifact manifest for completed work.
        
        Args:
            task_id: Associated task
            component_name: Name of the component implemented
            outputs: Paths to outputs (code, tests, docs)
            interface: API specification
            version: Semantic version
        
        Returns:
            Artifact dictionary
        """
        artifact = {
            "task_id": task_id,
            "component": component_name,
            "version": version,
            "workstream": self.workstream,
            "agent": self.name,
            "created_at": datetime.now().isoformat(),
            "outputs": outputs,
            "interface": interface,
            "validation_status": "pending",
            "reviewer": f"reviewer_{self.workstream}"
        }
        
        # Write artifact manifest
        artifact_dir = self.output_dir / component_name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_path = artifact_dir / "artifact.json"
        with open(artifact_path, 'w') as f:
            json.dump(artifact, f, indent=2)
        
        self.logger.info(f"Created artifact for {component_name} at {artifact_path}")
        
        return artifact
    
    def request_review(self, artifact_path: Path) -> bool:
        """
        Trigger reviewer agent for this artifact.
        
        Args:
            artifact_path: Path to artifact.json
        
        Returns:
            True if review was successfully requested
        """
        with open(artifact_path) as f:
            artifact = json.load(f)
        
        # Create review request
        review_dir = Path("handoffs/reviews")
        review_dir.mkdir(parents=True, exist_ok=True)
        
        review_request = {
            "artifact_path": str(artifact_path),
            "task_id": artifact["task_id"],
            "component": artifact["component"],
            "reviewer": artifact["reviewer"],
            "requested_by": self.name,
            "requested_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        review_path = review_dir / f"{artifact['task_id']}_review_request.json"
        with open(review_path, 'w') as f:
            json.dump(review_request, f, indent=2)
        
        self.logger.info(f"Requested review for {artifact['component']}")
        return True
    
    def complete_task(self, task_id: str, artifact: Dict[str, Any]):
        """
        Mark task as complete and create all necessary handoff documents.
        
        Args:
            task_id: Task that was completed
            artifact: Artifact manifest dictionary
        """
        # Update task status
        self.update_task_status(
            task_id, 
            TaskStatus.REVIEW,
            {"artifact_path": artifact["outputs"]}
        )
        
        # Create handoff documents for downstream dependencies
        task = self.load_task(task_id)
        self._create_handoffs(task, artifact)
        
        # Request review
        artifact_path = Path(self.output_dir) / artifact["component"] / "artifact.json"
        self.request_review(artifact_path)
        
        self.logger.info(f"Completed task {task_id}, status: REVIEW")
    
    def _create_handoffs(self, task: Dict[str, Any], artifact: Dict[str, Any]):
        """Create handoff documents for downstream consumers"""
        handoff_dir = Path("handoffs")
        handoff_dir.mkdir(exist_ok=True)
        
        # Determine downstream consumers from task metadata
        downstream = task.get("downstream_consumers", [])
        
        for consumer in downstream:
            handoff = {
                "from_agent": self.name,
                "from_task": task["id"],
                "to_agent": consumer,
                "artifact": artifact,
                "created_at": datetime.now().isoformat(),
                "message": f"Component {artifact['component']} is ready for integration",
                "integration_notes": task.get("integration_notes", "")
            }
            
            handoff_path = handoff_dir / f"{task['id']}_to_{consumer}.json"
            with open(handoff_path, 'w') as f:
                json.dump(handoff, f, indent=2)
            
            self.logger.info(f"Created handoff to {consumer}")
    
    def load_dependencies(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Load artifacts from dependency tasks.
        
        Args:
            task: Current task specification
        
        Returns:
            List of artifact dictionaries from dependencies
        """
        dependencies = task.get("dependencies", [])
        artifacts = []
        
        for dep_id in dependencies:
            # Find handoff document
            handoff_pattern = f"{dep_id}_to_{self.workstream}*.json"
            handoff_dir = Path("handoffs")
            
            handoff_files = list(handoff_dir.glob(handoff_pattern))
            
            if not handoff_files:
                self.logger.warning(f"No handoff found for dependency {dep_id}")
                continue
            
            # Load the handoff
            with open(handoff_files[0]) as f:
                handoff = json.load(f)
            
            artifacts.append(handoff["artifact"])
            self.logger.info(f"Loaded dependency artifact from {dep_id}")
        
        return artifacts
    
    def run(self, task_id: str):
        """
        Main entry point for agent execution.
        
        Args:
            task_id: ID of task to execute
        """
        try:
            # Load task
            task = self.load_task(task_id)
            
            # Update status to in_progress
            self.update_task_status(task_id, TaskStatus.IN_PROGRESS)
            
            # Execute task (implemented by subclass)
            self.logger.info(f"Starting execution of task {task_id}")
            artifact = self.execute_task(task)
            
            # Complete task
            self.complete_task(task_id, artifact)
            
            self.logger.info(f"Successfully completed task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute task {task_id}: {str(e)}", exc_info=True)
            self.update_task_status(
                task_id, 
                TaskStatus.FAILED,
                {"error": str(e)}
            )
            raise
