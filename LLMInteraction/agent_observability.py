"""
Multi-Agent Systems: Complete Learning Guide
============================================

This file teaches multi-agent systems through practical examples. Run it directly
to see each pattern in action. Each section is self-contained with detailed
comments explaining the concepts.

Topics Covered:
1. Supervisor Pattern - One agent coordinates others
2. Hierarchical Agents - Chain of command with escalation
3. Swarm Architectures - Decentralized, emergent behavior
4. Collaborative Agents - Agents working together on a task
5. Planner-Executor-Reviewer Pattern - Plan, do, check cycle
6. Critic Agent - Self-improvement through critique
7. Voting & Consensus - Democratic decision making
8. Negotiation & Delegation - Agents bargaining and assigning work
9. Agent Communication - Structured message passing
10. Shared Memory - Agents sharing knowledge
11. Multi-Agent Debate - Agents arguing different positions
12. Ensemble Agents - Combining multiple models
13. Router Agent - Intelligent task routing

Requirements:
pip install langgraph langchain_google_genai python-dotenv langchain_core pydantic
"""

import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Literal, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, FunctionMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# Configuration
LLM_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =============================================================================
# SECTION 0: UTILITY FUNCTIONS & BASE CLASSES
# =============================================================================

class AgentRole(Enum):
    """Roles an agent can play in a multi-agent system"""
    SUPERVISOR = "supervisor"
    WORKER = "worker"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    CRITIC = "critic"
    NEGOTIATOR = "negotiator"
    VOTER = "voter"
    DELEGATOR = "delegator"
    SPECIALIST = "specialist"
    ROUTER = "router"

@dataclass
class AgentMessage:
    """Standard message format for agent communication"""
    sender: str
    receiver: str
    content: str
    message_type: Literal["request", "response", "broadcast", "command", "question"]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

class SharedMemory:
    """Shared memory for agents to store and retrieve information"""
    def __init__(self):
        self._memory: Dict[str, Any] = {}
        self._history: List[AgentMessage] = []
        self._knowledge_base: Dict[str, str] = {}
    
    def store(self, key: str, value: Any) -> None:
        self._memory[key] = value
        print(f"[shared_memory] Stored: {key} = {str(value)[:50]}...")
    
    def retrieve(self, key: str) -> Optional[Any]:
        return self._memory.get(key)
    
    def add_message(self, message: AgentMessage) -> None:
        self._history.append(message)
    
    def get_history(self) -> List[AgentMessage]:
        return self._history
    
    def add_knowledge(self, topic: str, content: str) -> None:
        self._knowledge_base[topic] = content
        print(f"[shared_memory] Added knowledge: {topic}")
    
    def get_knowledge(self, topic: str) -> Optional[str]:
        return self._knowledge_base.get(topic)
    
    def get_all_knowledge(self) -> Dict[str, str]:
        return self._knowledge_base

# Global shared memory instance
shared_memory = SharedMemory()

# =============================================================================
# SECTION 1: SUPERVISOR PATTERN
# 
# A single "supervisor" agent coordinates multiple worker agents. The supervisor
# decides which worker to assign tasks to, collects their results, and compiles
# the final output. This is the most common multi-agent pattern.
# =============================================================================

class SupervisorPattern:
    """
    SUPERVISOR PATTERN:
    One agent (supervisor) coordinates multiple specialized worker agents.
    
    Key Concepts:
    - Centralized control
    - Clear hierarchy
    - Easy to manage
    - Single point of failure
    
    Use Case: Task routing, project management, orchestration
    """
    
    @staticmethod
    def demo():
        print("\n" + "="*60)
        print("SECTION 1: SUPERVISOR PATTERN")
        print("="*60)
        print("A supervisor agent coordinates specialized workers")
        print("-"*60)
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
        
        # Define worker agents as functions
        def code_worker(task: str) -> str:
            prompt = f"""You are a code expert. Given this task: '{task}'
            Provide 2-3 bullet points on what code changes are needed."""
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        def documentation_worker(task: str) -> str:
            prompt = f"""You are a documentation expert. Given this task: '{task}'
            Provide 2-3 bullet points on what documentation updates are needed."""
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        def testing_worker(task: str) -> str:
            prompt = f"""You are a QA testing expert. Given this task: '{task}'
            Provide 2-3 bullet points on what tests are needed."""
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        # Supervisor agent - decides who should work on what
        def supervisor(task: str) -> Dict[str, str]:
            supervisor_prompt = f"""You are a supervisor agent. Analyze this task: '{task}'
            
            Determine which specialists should work on it. Choose from:
            - code_worker: for programming, algorithms, or software changes
            - documentation_worker: for writing, guides, or documentation
            - testing_worker: for quality assurance, testing, or validation
            
            Return a JSON dictionary with:
            - 'workers': list of worker names to assign
            - 'reasoning': brief explanation of your choice
            
            Example: {{"workers": ["code_worker", "testing_worker"], "reasoning": "..."}}
            """
            
            response = llm.invoke([HumanMessage(content=supervisor_prompt)])
            
            # Simple parsing - in production, use structured output
            import re
            workers_match = re.search(r'"workers":\s*\[(.*?)\]', response.content)
            if workers_match:
                workers_str = workers_match.group(1)
                workers = [w.strip().strip('"\'') for w in workers_str.split(',')]
            else:
                # Fallback: assign all workers
                workers = ["code_worker", "documentation_worker", "testing_worker"]
            
            # Execute workers
            results = {}
            worker_functions = {
                "code_worker": code_worker,
                "documentation_worker": documentation_worker,
                "testing_worker": testing_worker
            }
            
            for worker_name in workers:
                if worker_name in worker_functions:
                    print(f"[supervisor] Assigning task to {worker_name}...")
                    results[worker_name] = worker_functions[worker_name](task)
                    print(f"[{worker_name}] Completed")
            
            return results
        
        # Demonstrate the supervisor pattern
        task = "We need to add a 'dark mode' feature to our web application"
        print(f"\nTask: {task}\n")
        
        results = supervisor(task)
        
        print("\nResults from supervisor-coordinated workers:")
        for worker, result in results.items():
            print(f"\n[{worker.upper()}]:")
            print(result)
        
        # Store results in shared memory
        shared_memory.store("supervisor_task", task)
        shared_memory.store("supervisor_results", results)
        
        print("\n✅ Supervisor Pattern Complete!")
        print("The supervisor orchestrated multiple specialized workers.")

# =============================================================================
# SECTION 2: HIERARCHICAL AGENTS
# 
# Agents are organized in a hierarchy (Manager -> Senior -> Junior).
# Tasks flow down the hierarchy, with higher-level agents handling
# more complex decisions and delegating subtasks.
# =============================================================================

class HierarchicalAgents:
    """
    HIERARCHICAL AGENTS:
    Agents organized in a chain of command with escalation.
    
    Key Concepts:
    - Manager: Strategic decisions, task delegation
    - Senior: Tactical decisions, problem solving
    - Junior: Execution, data collection
    
    Use Case: Complex problem solving, organizational simulation, escalation
    """
    
    @staticmethod
    def demo():
        print("\n" + "="*60)
        print("SECTION 2: HIERARCHICAL AGENTS")
        print("="*60)
        print("Manager -> Senior -> Junior hierarchy for task execution")
        print("-"*60)
        
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
        
        class AgentLevel:
            JUNIOR = "junior"
            SENIOR = "senior"
            MANAGER = "manager"
        
        # Junior agent - executes basic tasks
        def junior_agent(task: str) -> Dict[str, Any]:
            prompt = f"""You are a JUNIOR agent. You execute basic tasks and gather information.
            
            Task: {task}
            
            Provide:
            1. What information you can gather
            2. What basic analysis you can do
            3. What you need to escalate to a senior
            
            Keep it brief (2-3 sentences each)."""
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Determine if escalation needed
            escalation_keywords = ["complex", "difficult", "uncertain", "not sure", "requires expertise"]
            needs_escalation = any(kw in response.content.lower() for kw in escalation_keywords)
            
            return {
                "level": AgentLevel.JUNIOR,
                "analysis": response.content,
                "escalation_needed": needs_escalation,
                "escalation_reason": "Complexity detected" if needs_escalation else None
            }
        
        # Senior agent - handles complex tasks
        def senior_agent(task: str, junior_output: str = None) -> Dict[str, Any]:
            context = f"Junior agent analysis: {junior_output}\n\n" if junior_output else ""
            
            prompt = f"""{context}You are a SENIOR agent. You handle complex tasks and guide junior agents.
            
            Task: {task}
            
            Provide:
            1. Your analysis of the problem
            2. What guidance you'd give to the junior
            3. What needs to escalate to management
            
            Keep it brief (2-3 sentences each)."""
            response = llm.invoke([HumanMessage(content=prompt)])
            
            needs_escalation = any(kw in response.content.lower() 
                                  for kw in ["budget", "resources", "approval", "policy", "strategic"])
            
            return {
                "level": AgentLevel.SENIOR,
                "analysis": response.content,
                "escalation_needed": needs_escalation,
                "escalation_reason": "Strategic decision needed" if needs_escalation else None
            }
        
        # Manager agent - makes strategic decisions
        def manager_agent(task: str, junior_analysis: str = None, senior_analysis: str = None) -> Dict[str, Any]:
            context = ""
            if junior_analysis:
                context += f"Junior analysis: {junior_analysis}\n\n"
            if senior_analysis:
                context += f"Senior analysis: {senior_analysis}\n\n"
            
            prompt = f"""{context}You are a MANAGER agent. You make strategic decisions and allocate resources.
            
            Task: {task}
            
            Provide:
            1. Your strategic assessment
            2. Resource allocation recommendation
            3. Approval or rejection with reasoning
            
            Keep it brief (2-3 sentences each)."""
            response = llm.invoke([HumanMessage(content=prompt)])
            
            return {
                "level": AgentLevel.MANAGER,
                "decision": response.content,
                "approved": "approve" in response.content.lower()
            }
        
        # Demonstrate the hierarchy
        task = "We need to redesign our customer onboarding flow to reduce drop-off by 20%"
        print(f"\nTask: {task}\n")
        
        print("1️⃣ JUNIOR AGENT - Gathering basic information...")
        junior_result = junior_agent(task)
        print(f"[Junior] {junior_result['analysis'][:200]}...")
        print(f"Escalation needed: {junior_result['escalation_needed']}")
        
        if junior_result["escalation_needed"]:
            print("\n2️⃣ SENIOR AGENT - Handling escalated issue...")
            senior_result = senior_agent(task, junior_result["analysis"])
            print(f"[Senior] {senior_result['analysis'][:200]}...")
            print(f"Escalation needed: {senior_result['escalation_needed']}")
            
            if senior_result["escalation_needed"]:
                print("\n3️⃣ MANAGER AGENT - Making strategic decision...")
                manager_result = manager_agent(task, junior_result["analysis"], senior_result["analysis"])
                print(f"[Manager] {manager_result['decision'][:200]}...")
                print(f"Approved: {manager_result['approved']}")
        
        # Store in shared memory
        shared_memory.store("hierarchical_task", task)
        shared_memory.store("hierarchical_results", {
            "junior": junior_result,
            "senior": senior_result if junior_result["escalation_needed"] else None,
            "manager": manager_result if (junior_result["escalation_needed"] and 
                                         senior_result.get("escalation_needed", False)) else None
        })
        
        print("\n✅ Hierarchical Agents Pattern Complete!")
        print("Tasks escalated from Junior -> Senior -> Manager as needed.")

# =============================================================================
# SECTION 3: SWARM ARCHITECTURES
# 
# Decentralized agents working together without a central controller.
# Each agent is autonomous but follows simple rules. Emergent behavior
# arises from their interactions.
# =============================================================================

class SwarmArchitecture:
    """
    SWARM ARCHITECTURE:
    Decentralized agents with no central controller. Emergent behavior
    from local interactions.
    
    Key Concepts:
    - No central control
    - Local interactions only
    - Emergent behavior
    - Scalable
    - Fault tolerant
    
    Use Case: Distributed problem solving, self-organizing systems, simulation
    """
    
    @staticmethod
    def demo():
        print("\n" + "="*60)
        print("SECTION 3: SWARM ARCHITECTURE")
        print("="*60)
        print("Decentralized agents working together without a supervisor")
        print("-"*60)
        
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.5)
        
        class SwarmAgent:
            """Individual agent in the swarm"""
            def __init__(self, agent_id: str, specialty: str, llm):
                self.id = agent_id
                self.specialty = specialty
                self.llm = llm
                self.messages = []
                
            def think(self, context: str) -> str:
                """Agent thinks about the task and generates response"""
                prompt = f"""You are a swarm agent with ID {self.id}, specialty: {self.specialty}.
                
                Context: {context}
                
                Provide your perspective on this problem. Keep it brief (2-3 sentences)."""
                response = self.llm.invoke([HumanMessage(content=prompt)])
                self.messages.append(("think", response.content))
                return response.content
            
            def respond_to_peer(self, message: str) -> str:
                """Agent responds to another agent's message"""
                prompt = f"""You are a swarm agent with ID {self.id}, specialty: {self.specialty}.
                
                Another agent said: {message}
                
                How do you respond? Provide your perspective (2-3 sentences)."""
                response = self.llm.invoke([HumanMessage(content=prompt)])
                self.messages.append(("respond", response.content))
                return response.content
            
            def vote(self, options: list) -> str:
                """Agent votes on an option"""
                prompt = f"""You are a swarm agent with ID {self.id}, specialty: {self.specialty}.
                
                Vote on these options: {options}
                
                Choose one and explain why (1-2 sentences)."""
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
        
        # Create swarm
        specialties = ["frontend", "backend", "database", "security", "performance"]
        swarm = [SwarmAgent(f"agent_{i}", spec, llm) for i, spec in enumerate(specialties)]
        
        # Simulate swarm thinking
        problem = "How can we reduce API response time from 500ms to under 100ms?"
        print(f"\nProblem: {problem}")
        
        print("\n🔄 Swarm agents thinking independently...")
        
        # Phase 1: Independent thought
        responses = {}
        for agent in swarm:
            print(f"  Agent {agent.id} ({agent.specialty}) thinking...")
            response = agent.think(problem)
            responses[agent.id] = response
            print(f"    → {response[:100]}...")
        
        print("\n💬 Swarm agents sharing ideas...")
        
        # Phase 2: Peer sharing (each agent sees 2 random other responses)
        import random
        for agent in swarm:
            others = [a for a in swarm if a.id != agent.id]
            selected = random.sample(others, min(2, len(others)))
            
            for other in selected:
                print(f"  {agent.id} responding to {other.id}:")
                response = agent.respond_to_peer(responses[other.id])
                print(f"    → {response[:100]}...")
        
        # Phase 3: Voting
        options = ["option_a", "option_b", "option_c"]
        print("\n📊 Swarm voting on solutions...")
        
        votes = {}
        for agent in swarm:
            vote = agent.vote(options)
            print(f"  {agent.id}: {vote}")
            votes[agent.id] = vote
        
        # Aggregate results
        option_counts = {opt: 0 for opt in options}
        for vote in votes.values():
            for opt in options:
                if opt.lower() in vote.lower():
                    option_counts[opt] += 1
        
        print("\n📊 Voting results:")
        for opt, count in option_counts.items():
            print(f"  {opt}: {count} votes")
        
        winner = max(option_counts, key=option_counts.get)
        print(f"\nWinner: {winner} with {option_counts[winner]} votes")
        
        # Store in shared memory
        shared_memory.store("swarm_problem", problem)
        shared_memory.store("swarm_results", {
            "responses": responses,
            "votes": votes,
            "winner": winner
        })
        
        print("\n✅ Swarm Architecture Complete!")
        print("Decentralized agents reached a consensus through emergent behavior.")

# =============================================================================
# SECTION 4: COLLABORATIVE AGENTS
# 
# Agents work together on a shared task, combining their unique perspectives
# and capabilities. Unlike the supervisor pattern, collaboration is more
# peer-to-peer with shared ownership.
# =============================================================================

class CollaborativeAgents:
    """
    COLLABORATIVE AGENTS:
    Peers working together on a shared task. Each contributes their
    unique perspective.
    
    Key Concepts:
    - Equal status
    - Shared ownership
    - Complementary skills
    - Peer-to-peer communication
    
    Use Case: Creative tasks, problem solving, research
    """
    
    @staticmethod
    def demo():
        print("\n" + "="*60)
        print("SECTION 4: COLLABORATIVE AGENTS")
        print("="*60)
        print("Peer agents collaborate on a shared goal")
        print("-"*60)
        
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.4)
        
        # Define specialized agents
        def researcher(task: str) -> str:
            prompt = f"""You are a RESEARCHER agent. You gather information and data.
            
            Task context: {task}
            
            Provide:
            1. What information would be valuable
            2. What data sources might help
            3. Key facts to consider"""
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        def designer(task: str) -> str:
            prompt = f"""You are a DESIGNER agent. You focus on user experience and aesthetics.
            
            Task context: {task}
            
            Provide:
            1. User experience considerations
            2. Design patterns that might apply
            3. Visual or interaction suggestions"""
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        def engineer(task: str) -> str:
            prompt = f"""You are an ENGINEER agent. You focus on implementation and feasibility.
            
            Task context: {task}
            
            Provide:
            1. Technical considerations
            2. Implementation approach
            3. Potential technical challenges"""
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        def product_manager(task: str, researcher_input: str, designer_input: str, engineer_input: str) -> str:
            prompt = f"""You are a PRODUCT MANAGER agent. You synthesize team input into a plan.
            
            Task: {task}
            
            Researcher said: {researcher_input}
            Designer said: {designer_input}
            Engineer said: {engineer_input}
            
            Synthesize these inputs into a cohesive recommendation.
            Provide 3 bullet points summarizing the path forward."""
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        # Demonstrate collaboration
        task = "Create a new mobile app for tracking personal fitness goals"
        print(f"\nTask: {task}\n")
        
        print("🤝 Agents collaborating...")
        
        # Stage 1: Independent research
        print("\n📚 Researcher gathering information...")
        research_input = researcher(task)
        print(f"[Researcher]\n{research_input[:200]}...")
        
        print("\n🎨 Designer exploring user experience...")
        design_input = designer(task)
        print(f"[Designer]\n{design_input[:200]}...")
        
        print("\n⚙️ Engineer evaluating implementation...")
        engineering_input = engineer(task)
        print(f"[Engineer]\n{engineering_input[:200]}...")
        
        # Stage 2: Synthesis
        print("\n📋 Product Manager synthesizing...")
        final_plan = product_manager(task, research_input, design_input, engineering_input)
        print(f"[Product Manager]\n{final_plan}")
        
        # Store in shared memory
        shared_memory.store("collaboration_task", task)
        shared_memory.store("collaboration_results", {
            "researcher": research_input,
            "designer": design_input,
            "engineer": engineering_input,
            "product_manager": final_plan
        })
        
        print("\n✅ Collaborative Agents Pattern Complete!")
        print("Multiple agents contributed unique perspectives to create a unified solution.")

# =============================================================================
# SECTION 5: PLANNER-EXECUTOR-REVIEWER PATTERN
# 
# Three specialized agents: Planner creates a plan, Executor implements it,
# and Reviewer evaluates the results. This creates a "Plan-Do-Check" cycle.
# =============================================================================

class PlannerExecutorReviewer:
    """
    PLANNER-EXECUTOR-REVIEWER PATTERN:
    Three specialized agents in a "Plan-Do-Check" cycle.
    
    Key Concepts:
    - Planner: Strategic planning, breaking down tasks
    - Executor: Implementation, following the plan
    - Reviewer: Evaluation, quality control
    
    Use Case: Complex projects, iterative development, quality assurance
    """
    
    @staticmethod
    def demo():
        print("\n" + "="*60)
        print("SECTION 5: PLANNER-EXECUTOR-REVIEWER PATTERN")
        print("="*60)
        print("Plan → Do → Check cycle with specialized agents")
        print("-"*60)
        
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
        
        class Plan(BaseModel):
            """Structured plan from the planner"""
            goal: str
            steps: List[str]
            success_criteria: List[str]
            estimated_effort: str
            dependencies: List[str] = Field(default_factory=list)
        
        class ExecutionResult(BaseModel):
            """Result from the executor"""
            completed_steps: List[str]
            output: str
            issues_encountered: List[str]
            time_taken: str
        
        class ReviewResult(BaseModel):
            """Review from the reviewer"""
            score: int = Field(ge=1, le=10)
            strengths: List[str]
            weaknesses: List[str]
            recommendations: List[str]
            approved: bool
        
        # Planner Agent
        def planner_agent(task: str) -> Plan:
            parser = PydanticOutputParser(pydantic_object=Plan)
            
            prompt = f"""You are a PLANNER agent. Create a detailed plan for:
            
            Task: {task}
            
            {parser.get_format_instructions()}
            
            Be specific and actionable."""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            try:
                plan = parser.parse(response.content)
                print(f"[Planner] Created plan with {len(plan.steps)} steps")
                return plan
            except:
                # Fallback plan
                return Plan(
                    goal=task,
                    steps=["Analyze requirements", "Design solution", "Implement", "Test", "Deploy"],
                    success_criteria=["All requirements met", "No critical bugs"],
                    estimated_effort="Medium"
                )
        
        # Executor Agent
        def executor_agent(plan: Plan) -> ExecutionResult:
            prompt = f"""You are an EXECUTOR agent. Implement this plan:
            
            Plan:
            Goal: {plan.goal}
            Steps: {', '.join(plan.steps)}
            Success Criteria: {', '.join(plan.success_criteria)}
            
            Provide execution results including:
            1. What was completed
            2. Output produced
            3. Issues encountered
            4. Time taken (estimate)"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Parse response into structured result
            return ExecutionResult(
                completed_steps=plan.steps[:3],  # Simulate partial completion
                output=response.content,
                issues_encountered=["Minor configuration issue"],
                time_taken="2 hours"
            )
        
        # Reviewer Agent
        def reviewer_agent(task: str, plan: Plan, result: ExecutionResult) -> ReviewResult:
            prompt = f"""You are a REVIEWER agent. Evaluate this execution:
            
            Original Task: {task}
            
            Plan:
            Goal: {plan.goal}
            Steps: {', '.join(plan.steps)}
            Success Criteria: {', '.join(plan.success_criteria)}
            
            Execution Result:
            Completed: {', '.join(result.completed_steps)}
            Output: {result.output[:500]}
            Issues: {', '.join(result.issues_encountered)}
            
            Provide a review with:
            - Score (1-10)
            - Strengths
            - Weaknesses
            - Recommendations
            - Approval status (approved/rejected)"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Simple parsing for demo
            score = 7
            approved = "approved" in response.content.lower()
            
            return ReviewResult(
                score=score,
                strengths=["Good initial approach", "Clear documentation"],
                weaknesses=["Some steps not completed", "Missing edge cases"],
                recommendations=["Complete remaining steps", "Add error handling"],
                approved=approved
            )
        
        # Demonstrate the pattern
        task = "Build a REST API for a todo list application"
        print(f"\nTask: {task}\n")
        
        # Step 1: Plan
        print("📋 PLANNER: Creating plan...")
        plan = planner_agent(task)
        print(f"Goal: {plan.goal}")
        print(f"Steps: {', '.join(plan.steps)}")
        print(f"Success Criteria: {', '.join(plan.success_criteria)}")
        
        # Step 2: Execute
        print("\n⚙️ EXECUTOR: Implementing plan...")
        result = executor_agent(plan)
        print(f"Completed: {', '.join(result.completed_steps)}")
        print(f"Issues: {', '.join(result.issues_encountered)}")
        
        # Step 3: Review
        print("\n🔍 REVIEWER: Evaluating execution...")
        review = reviewer_agent(task, plan, result)
        print(f"Score: {review.score}/10")
        print(f"Strengths: {', '.join(review.strengths)}")
        print(f"Weaknesses: {', '.join(review.weaknesses)}")
        print(f"Recommendations: {', '.join(review.recommendations)}")
        print(f"Approved: {review.approved}")
        
        # Store in shared memory
        shared_memory.store("per_task", task)
        shared_memory.store("per_plan", plan)
        shared_memory.store("per_result", result)
        shared_memory.store("per_review", review)
        
        print("\n✅ Planner-Executor-Reviewer Pattern Complete!")
        print("Plan → Execute → Review cycle demonstrated.")

# =============================================================================
# SECTION 6: CRITIC AGENT
# 
# A critic agent evaluates and provides feedback to improve other agents'
# outputs. This is essential for self-improvement and quality assurance.
# =============================================================================

class CriticAgent:
    """
    CRITIC AGENT:
    An agent that evaluates outputs and provides constructive feedback.
    
    Key Concepts:
    - Self-improvement through critique
    - Quality assurance
    - Iterative refinement
    - Constructive feedback
    
    Use Case: Content review, code review, quality control, self-improvement
    """
    
    @staticmethod
    def demo():
        print("\n" + "="*60)
        print("SECTION 6: CRITIC AGENT")
        print("="*60)
        print("An agent that critiques and improves other agents' work")
        print("-"*60)
        
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
        
        # Original generator agent
        def generator_agent(topic: str) -> str:
            prompt = f"""You are a CONTENT GENERATOR agent.
            
            Write a short paragraph (3-4 sentences) about: {topic}
            
            Make it informative but don't worry about perfection."""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content
        
        # Critic agent - provides constructive feedback
        def critic_agent(content: str, topic: str) -> Dict[str, Any]:
            prompt = f"""You are a CRITIC agent. Provide constructive feedback on this content:
            
            Topic: {topic}
            
            Content:
            {content}
            
            Provide:
            1. Score (1-10)
            2. 2-3 specific strengths
            3. 2-3 specific areas for improvement
            4. Specific suggestions for improvement
            5. Revised version that addresses the issues
            
            Be constructive and helpful."""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Simple parsing
            score = 7
            if "score" in response.content.lower():
                import re
                score_match = re.search(r'\b([1-9]|10)\b', response.content)
                if score_match:
                    score = int(score_match.group(0))
            
            return {
                "score": score,
                "feedback": response.content,
                "revised": "Revised version would be generated here"
            }
        
        # Iterative improvement agent
        def iterative_improver(topic: str, iterations: int = 2) -> str:
            print(f"\n📝 Generating initial content about: {topic}")
            content = generator_agent(topic)
            print(f"Initial version:\n{content}\n")
            
            for i in range(iterations):
                print(f"\n🔄 Iteration {i+1}: Critic reviewing...")
                critique = critic_agent(content, topic)
                print(f"Score: {critique['score']}/10")
                
                # Improve the content based on critique
                improvement_prompt = f"""You are a CONTENT GENERATOR agent.
                
                Original content: {content}
                
                Critic feedback: {critique['feedback'][:300]}
                
                Improve the content based on the feedback. Provide the improved version."""
                
                improved_response = llm.invoke([HumanMessage(content=improvement_prompt)])
                content = improved_response.content
                
                print(f"Improved version:\n{content}\n")
            
            return content
        
        # Demonstrate the critic agent
        topic = "The future of artificial intelligence"
        print(f"\nTopic: {topic}\n")
        
        print("🔄 Iterative improvement with critic feedback...")
        final_content = iterative_improver(topic, iterations=2)
        
        # Store in shared memory
        shared_memory.store("critic_topic", topic)
        shared_memory.store("critic_final_content", final_content)
        
        print("\n✅ Critic Agent Pattern Complete!")
        print("The critic agent provided feedback that improved the content quality.")

# =============================================================================
# SECTION 7: VOTING AND CONSENSUS
# 
# Multiple agents vote on decisions, and consensus is reached through
# democratic processes. This is useful for reducing bias and improving
# decision quality.
# =============================================================================

class VotingAndConsensus:
    """
    VOTING AND CONSENSUS:
    Multiple agents vote on decisions to reach consensus.
    
    Key Concepts:
    - Democratic decision making
    - Bias reduction
    - Multiple perspectives
    - Weighted voting
    - Consensus mechanisms
    
    Use Case: Decision making, classification, ensemble methods, governance
    """
    
    @staticmethod
    def demo():
        print("\n" + "="*60)
        print("SECTION 7: VOTING AND CONSENSUS")
        print("="*60)
        print("Agents vote to reach consensus on decisions")
        print("-"*60)
        
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.5)
        
        # Create voting agents with different specialties
        voting_agents = [
            {"name": "Tech Expert", "specialty": "Technical feasibility"},
            {"name": "Business Analyst", "specialty": "Business value"},
            {"name": "UX Designer", "specialty": "User experience"},
            {"name": "Security Specialist", "specialty": "Security and compliance"},
            {"name": "Product Manager", "specialty": "Product strategy"}
        ]
        
        # Each agent votes
        def agent_vote(agent_info: Dict, question: str, options: List[str]) -> Dict[str, Any]:
            prompt = f"""You are a {agent_info['name']}. Your specialty: {agent_info['specialty']}
            
            Question: {question}
            
            Options:
            {', '.join(options)}
            
            Vote for ONE option. Provide:
            1. Your choice
            2. Brief reasoning (1-2 sentences)
            3. Confidence (1-5)"""
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            # Determine choice
            choice = options[0]  # Default
            for option in options:
                if option.lower() in response.content.lower():
                    choice = option
                    break
            
            return {
                "agent": agent_info["name"],
                "specialty": agent_info["specialty"],
                "choice": choice,
                "reasoning": response.content,
                "confidence": 4  # Default confidence
            }
        
        # Demonstrate