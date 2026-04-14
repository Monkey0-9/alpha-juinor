#!/usr/bin/env python3
"""
TEAM STRUCTURE AND EXPERTISE FRAMEWORK
======================================

Define the organizational structure and expertise requirements for a top 1% trading firm.
This provides the blueprint for building the human capital needed for institutional success.

Features:
- Role definitions and responsibilities
- Expertise requirements and qualifications
- Team collaboration frameworks
- Knowledge sharing systems
- Performance evaluation metrics
- Compensation structures
"""

import asyncio
import time
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TeamRole:
    """Team role definition"""
    title: str
    department: str
    level: str  # junior, senior, lead, principal, partner
    responsibilities: List[str]
    
    # Requirements
    education: List[str]
    experience_years: int
    certifications: List[str]
    technical_skills: List[str]
    
    # Compensation
    base_salary_range: Tuple[int, int]  # (min, max) in USD
    bonus_percentage: float
    equity_percentage: float
    
    # Performance metrics
    kpis: List[str]
    success_criteria: List[str]


@dataclass
class TeamMember:
    """Team member profile"""
    employee_id: str
    name: str
    role: TeamRole
    start_date: datetime = field(default_factory=datetime.utcnow)
    
    # Performance
    performance_score: float = 0.0  # 0-5
    bonus_earned: float = 0.0
    equity_vested: float = 0.0
    
    # Status
    is_active: bool = True
    last_review: Optional[datetime] = None
    next_review: Optional[datetime] = None


@dataclass
class TeamCollaboration:
    """Team collaboration configuration"""
    teams: List[str]
    collaboration_type: str  # cross_functional, project_based, hierarchical
    meeting_frequency: str
    communication_channels: List[str]
    
    # Metrics
    collaboration_score: float = 0.0
    project_success_rate: float = 0.0


class TeamStructure:
    """
    Define and manage the organizational structure for a top 1% trading firm.
    
    This provides the blueprint for building the human capital needed
    to compete with the best hedge funds and trading firms.
    """
    
    def __init__(self):
        # Team roles
        self.roles: Dict[str, TeamRole] = {}
        
        # Team members
        self.members: Dict[str, TeamMember] = {}
        
        # Team collaborations
        self.collaborations: Dict[str, TeamCollaboration] = {}
        
        # Organizational metrics
        self.metrics = {
            'total_employees': 0,
            'active_employees': 0,
            'average_performance': 0.0,
            'total_compensation': 0.0,
            'team_satisfaction': 0.0
        }
        
        # Initialize roles
        self._initialize_roles()
        self._initialize_collaborations()
        
        logger.info("Team Structure initialized")
    
    def _initialize_roles(self):
        """Initialize all team roles for a top 1% trading firm"""
        
        # QUANTITATIVE RESEARCH TEAM
        
        # Head of Quant Research
        self.roles['head_quant_research'] = TeamRole(
            title='Head of Quantitative Research',
            department='Quantitative Research',
            level='partner',
            responsibilities=[
                'Lead quantitative research strategy',
                'Manage team of researchers and analysts',
                'Oversee model development and validation',
                'Collaborate with portfolio management',
                'Ensure research quality and innovation',
                'Mentor junior researchers'
            ],
            education=['PhD in Mathematics, Physics, Computer Science, or Economics'],
            experience_years=15,
            certifications=['CFA', 'FRM'],
            technical_skills=[
                'Python', 'R', 'MATLAB', 'C++', 'Machine Learning',
                'Statistical Modeling', 'Time Series Analysis',
                'Portfolio Theory', 'Risk Management'
            ],
            base_salary_range=(400000, 600000),
            bonus_percentage=0.5,
            equity_percentage=0.15,
            kpis=[
                'Research quality metrics',
                'Model performance',
                'Team productivity',
                'Innovation contributions'
            ],
            success_criteria=[
                'Consistent alpha generation',
                'Low model turnover',
                'High team retention',
                'Industry recognition'
            ]
        )
        
        # Senior Quant Researcher
        self.roles['senior_quant_researcher'] = TeamRole(
            title='Senior Quantitative Researcher',
            department='Quantitative Research',
            level='principal',
            responsibilities=[
                'Develop and validate trading models',
                'Conduct market research and analysis',
                'Collaborate with data science team',
                'Mentor junior researchers',
                'Present research findings to management'
            ],
            education=['PhD or Masters in Quantitative field'],
            experience_years=8,
            certifications=['CFA', 'FRM'],
            technical_skills=[
                'Python', 'R', 'MATLAB', 'Machine Learning',
                'Statistical Analysis', 'Data Mining',
                'Financial Modeling', 'Backtesting'
            ],
            base_salary_range=(250000, 350000),
            bonus_percentage=0.4,
            equity_percentage=0.08,
            kpis=[
                'Model performance',
                'Research productivity',
                'Collaboration score',
                'Innovation metrics'
            ],
            success_criteria=[
                'Consistent alpha generation',
                'Low model decay',
                'Successful model deployment',
                'Team contributions'
            ]
        )
        
        # Quant Researcher
        self.roles['quant_researcher'] = TeamRole(
            title='Quantitative Researcher',
            department='Quantitative Research',
            level='senior',
            responsibilities=[
                'Develop trading algorithms',
                'Conduct market analysis',
                'Backtest and validate models',
                'Assist in research projects',
                'Document research findings'
            ],
            education=['Masters or PhD in Quantitative field'],
            experience_years=4,
            certifications=['CFA', 'FRM'],
            technical_skills=[
                'Python', 'R', 'SQL', 'Machine Learning',
                'Statistical Analysis', 'Data Visualization',
                'Financial Markets', 'Backtesting'
            ],
            base_salary_range=(150000, 200000),
            bonus_percentage=0.3,
            equity_percentage=0.04,
            kpis=[
                'Model development',
                'Research output',
                'Quality metrics',
                'Team collaboration'
            ],
            success_criteria=[
                'Successful model deployment',
                'Research quality',
                'Team integration',
                'Skill development'
            ]
        )
        
        # DATA SCIENCE TEAM
        
        # Head of Data Science
        self.roles['head_data_science'] = TeamRole(
            title='Head of Data Science',
            department='Data Science',
            level='partner',
            responsibilities=[
                'Lead data science strategy',
                'Manage data infrastructure',
                'Oversee ML model development',
                'Collaborate with quant research',
                'Ensure data quality and governance',
                'Drive innovation in data analytics'
            ],
            education=['PhD in Computer Science, Statistics, or related field'],
            experience_years=12,
            certifications=['AWS Certified ML Specialist', 'Google Cloud ML Engineer'],
            technical_skills=[
                'Python', 'TensorFlow', 'PyTorch', 'Spark', 'Kubernetes',
                'Deep Learning', 'Big Data', 'Cloud Computing',
                'Data Engineering', 'MLOps'
            ],
            base_salary_range=(350000, 500000),
            bonus_percentage=0.45,
            equity_percentage=0.12,
            kpis=[
                'Data infrastructure quality',
                'ML model performance',
                'Team productivity',
                'Innovation metrics'
            ],
            success_criteria=[
                'Robust data pipelines',
                'High-quality ML models',
                'Scalable infrastructure',
                'Team excellence'
            ]
        )
        
        # Senior Data Scientist
        self.roles['senior_data_scientist'] = TeamRole(
            title='Senior Data Scientist',
            department='Data Science',
            level='principal',
            responsibilities=[
                'Develop ML models for trading',
                'Build data pipelines',
                'Conduct alternative data analysis',
                'Optimize model performance',
                'Mentor junior data scientists'
            ],
            education=['PhD or Masters in Data Science, Computer Science, or Statistics'],
            experience_years=6,
            certifications=['AWS Certified ML Specialist', 'TensorFlow Developer'],
            technical_skills=[
                'Python', 'TensorFlow', 'PyTorch', 'Spark', 'SQL',
                'Deep Learning', 'NLP', 'Computer Vision',
                'Big Data', 'Cloud Computing'
            ],
            base_salary_range=(200000, 300000),
            bonus_percentage=0.35,
            equity_percentage=0.06,
            kpis=[
                'ML model performance',
                'Data pipeline quality',
                'Research contributions',
                'Team collaboration'
            ],
            success_criteria=[
                'High-performing models',
                'Robust data infrastructure',
                'Innovative solutions',
                'Team leadership'
            ]
        )
        
        # PORTFOLIO MANAGEMENT TEAM
        
        # Chief Investment Officer
        self.roles['cio'] = TeamRole(
            title='Chief Investment Officer',
            department='Portfolio Management',
            level='partner',
            responsibilities=[
                'Oversee investment strategy',
                'Manage portfolio risk',
                'Allocate capital across strategies',
                'Report to investors',
                'Lead investment committee',
                'Drive fund performance'
            ],
            education=['MBA or Masters in Finance', 'CFA'],
            experience_years=20,
            certifications=['CFA', 'CAIA'],
            technical_skills=[
                'Portfolio Management', 'Risk Management', 'Asset Allocation',
                'Investment Analysis', 'Financial Modeling', 'Regulatory Compliance',
                'Investor Relations', 'Performance Attribution'
            ],
            base_salary_range=(500000, 800000),
            bonus_percentage=0.6,
            equity_percentage=0.2,
            kpis=[
                'Fund performance',
                'Risk-adjusted returns',
                'Investor satisfaction',
                'Capital allocation efficiency'
            ],
            success_criteria=[
                'Consistent outperformance',
                'Low volatility',
                'Investor retention',
                'Industry recognition'
            ]
        )
        
        # Portfolio Manager
        self.roles['portfolio_manager'] = TeamRole(
            title='Portfolio Manager',
            department='Portfolio Management',
            level='principal',
            responsibilities=[
                'Manage specific trading strategies',
                'Execute investment decisions',
                'Monitor portfolio risk',
                'Optimize portfolio performance',
                'Collaborate with research team'
            ],
            education=['MBA or Masters in Finance', 'CFA'],
            experience_years=10,
            certifications=['CFA', 'CAIA'],
            technical_skills=[
                'Portfolio Management', 'Risk Management', 'Derivatives',
                'Fixed Income', 'Equity Analysis', 'Options Trading',
                'Performance Attribution', 'Risk Analytics'
            ],
            base_salary_range=(300000, 450000),
            bonus_percentage=0.5,
            equity_percentage=0.1,
            kpis=[
                'Strategy performance',
                'Risk management',
                'Capital efficiency',
                'Team collaboration'
            ],
            success_criteria=[
                'Consistent alpha generation',
                'Low drawdowns',
                'Capital efficiency',
                'Team integration'
            ]
        )
        
        # RISK MANAGEMENT TEAM
        
        # Chief Risk Officer
        self.roles['cro'] = TeamRole(
            title='Chief Risk Officer',
            department='Risk Management',
            level='partner',
            responsibilities=[
                'Oversee risk management framework',
                'Develop risk policies and procedures',
                'Monitor market and credit risk',
                'Ensure regulatory compliance',
                'Report to board and investors',
                'Lead risk committee'
            ],
            education=['MBA or Masters in Risk Management', 'CFA', 'FRM'],
            experience_years=15,
            certifications=['CFA', 'FRM', 'PRM'],
            technical_skills=[
                'Risk Management', 'Value at Risk', 'Stress Testing',
                'Regulatory Compliance', 'Credit Risk', 'Operational Risk',
                'Risk Analytics', 'Financial Modeling'
            ],
            base_salary_range=(350000, 500000),
            bonus_percentage=0.4,
            equity_percentage=0.12,
            kpis=[
                'Risk metrics accuracy',
                'Compliance score',
                'Risk event prevention',
                'Team effectiveness'
            ],
            success_criteria=[
                'No major risk events',
                'Regulatory compliance',
                'Effective risk monitoring',
                'Team excellence'
            ]
        )
        
        # Senior Risk Manager
        self.roles['senior_risk_manager'] = TeamRole(
            title='Senior Risk Manager',
            department='Risk Management',
            level='principal',
            responsibilities=[
                'Monitor portfolio risk metrics',
                'Conduct stress testing',
                'Analyze market risk exposure',
                'Implement risk controls',
                'Collaborate with trading team'
            ],
            education=['Masters in Risk Management, Finance', 'CFA', 'FRM'],
            experience_years=8,
            certifications=['CFA', 'FRM', 'PRM'],
            technical_skills=[
                'Risk Analytics', 'Value at Risk', 'Stress Testing',
                'Market Risk', 'Credit Risk', 'Operational Risk',
                'Risk Modeling', 'Financial Mathematics'
            ],
            base_salary_range=(200000, 300000),
            bonus_percentage=0.3,
            equity_percentage=0.06,
            kpis=[
                'Risk monitoring accuracy',
                'Stress test effectiveness',
                'Risk control implementation',
                'Team collaboration'
            ],
            success_criteria=[
                'Effective risk monitoring',
                'Accurate risk metrics',
                'Successful risk controls',
                'Team contributions'
            ]
        )
        
        # TRADING OPERATIONS TEAM
        
        # Head of Trading Operations
        self.roles['head_trading_operations'] = TeamRole(
            title='Head of Trading Operations',
            department='Trading Operations',
            level='principal',
            responsibilities=[
                'Oversee trade execution',
                'Manage trading systems',
                'Optimize execution algorithms',
                'Coordinate with brokers and exchanges',
                'Ensure trade compliance',
                'Manage operational risk'
            ],
            education=['MBA or Masters in Finance', 'CFA'],
            experience_years=12,
            certifications=['CFA', 'Series 7', 'Series 63'],
            technical_skills=[
                'Trading Systems', 'Order Management', 'Execution Algorithms',
                'Market Microstructure', 'Broker Relationships', 'Compliance',
                'Risk Management', 'Performance Analytics'
            ],
            base_salary_range=(250000, 350000),
            bonus_percentage=0.35,
            equity_percentage=0.08,
            kpis=[
                'Execution quality',
                'Cost optimization',
                'System reliability',
                'Compliance score'
            ],
            success_criteria=[
                'Low execution costs',
                'High execution quality',
                'System stability',
                'Regulatory compliance'
            ]
        )
        
        # Senior Trader
        self.roles['senior_trader'] = TeamRole(
            title='Senior Trader',
            department='Trading Operations',
            level='senior',
            responsibilities=[
                'Execute trading strategies',
                'Monitor market conditions',
                'Optimize trade execution',
                'Manage broker relationships',
                'Report trading performance'
            ],
            education=['Masters in Finance', 'CFA'],
            experience_years=6,
            certifications=['CFA', 'Series 7', 'Series 63'],
            technical_skills=[
                'Trading Execution', 'Market Analysis', 'Order Management',
                'Risk Management', 'Broker Relationships', 'Compliance',
                'Performance Analysis', 'Market Microstructure'
            ],
            base_salary_range=(150000, 250000),
            bonus_percentage=0.25,
            equity_percentage=0.04,
            kpis=[
                'Execution performance',
                'Cost reduction',
                'Market timing',
                'Risk management'
            ],
            success_criteria=[
                'Consistent execution quality',
                'Low trading costs',
                'Effective market timing',
                'Team collaboration'
            ]
        )
        
        # TECHNOLOGY TEAM
        
        # Chief Technology Officer
        self.roles['cto'] = TeamRole(
            title='Chief Technology Officer',
            department='Technology',
            level='partner',
            responsibilities=[
                'Lead technology strategy',
                'Oversee system architecture',
                'Manage technology team',
                'Ensure system security and reliability',
                'Drive innovation in fintech',
                'Align technology with business goals'
            ],
            education=['Masters or PhD in Computer Science', 'MBA'],
            experience_years=15,
            certifications=['AWS Solutions Architect', 'Google Cloud Professional'],
            technical_skills=[
                'System Architecture', 'Cloud Computing', 'Cybersecurity',
                'DevOps', 'Machine Learning', 'Big Data', 'Blockchain',
                'API Development', 'Database Management'
            ],
            base_salary_range=(400000, 600000),
            bonus_percentage=0.45,
            equity_percentage=0.15,
            kpis=[
                'System reliability',
                'Innovation metrics',
                'Team productivity',
                'Security posture'
            ],
            success_criteria=[
                'High system uptime',
                'Innovative solutions',
                'Team excellence',
                'Business alignment'
            ]
        )
        
        # Senior Software Engineer
        self.roles['senior_software_engineer'] = TeamRole(
            title='Senior Software Engineer',
            department='Technology',
            level='principal',
            responsibilities=[
                'Develop trading systems',
                'Implement algorithms',
                'Optimize system performance',
                'Ensure code quality',
                'Mentor junior engineers'
            ],
            education=['Masters in Computer Science', 'BS in Computer Science'],
            experience_years=8,
            certifications=['AWS Developer', 'Google Cloud Developer'],
            technical_skills=[
                'Python', 'C++', 'Java', 'Kubernetes', 'Docker',
                'Distributed Systems', 'API Development', 'Database Design',
                'Performance Optimization', 'Security'
            ],
            base_salary_range=(200000, 300000),
            bonus_percentage=0.3,
            equity_percentage=0.06,
            kpis=[
                'Code quality',
                'System performance',
                'Feature delivery',
                'Team collaboration'
            ],
            success_criteria=[
                'High-quality code',
                'System reliability',
                'Innovation',
                'Team leadership'
            ]
        )
        
        # COMPLIANCE TEAM
        
        # Chief Compliance Officer
        self.roles['cco'] = TeamRole(
            title='Chief Compliance Officer',
            department='Compliance',
            level='principal',
            responsibilities=[
                'Oversee compliance program',
                'Ensure regulatory compliance',
                'Develop compliance policies',
                'Conduct compliance training',
                'Manage regulatory relationships',
                'Report to board and regulators'
            ],
            education=['JD or Masters in Compliance', 'CAMS'],
            experience_years=12,
            certifications=['CAMS', 'CCEP', 'CRCM'],
            technical_skills=[
                'Regulatory Compliance', 'SEC/FINRA Rules', 'AML/KYC',
                'Risk Management', 'Audit Procedures', 'Legal Analysis',
                'Policy Development', 'Training Programs'
            ],
            base_salary_range=(250000, 400000),
            bonus_percentage=0.3,
            equity_percentage=0.08,
            kpis=[
                'Compliance score',
                'Regulatory relationships',
                'Training effectiveness',
                'Policy implementation'
            ],
            success_criteria=[
                'No regulatory violations',
                'Effective compliance program',
                'Strong regulator relationships',
                'Team excellence'
            ]
        )
        
        logger.info(f"Initialized {len(self.roles)} team roles")
    
    def _initialize_collaborations(self):
        """Initialize team collaboration structures"""
        
        # Quant Research + Data Science Collaboration
        self.collaborations['quant_data_science'] = TeamCollaboration(
            teams=['Quantitative Research', 'Data Science'],
            collaboration_type='cross_functional',
            meeting_frequency='weekly',
            communication_channels=['Slack', 'Jira', 'Confluence'],
            collaboration_score=0.0,
            project_success_rate=0.0
        )
        
        # Portfolio Management + Risk Management Collaboration
        self.collaborations['portfolio_risk'] = TeamCollaboration(
            teams=['Portfolio Management', 'Risk Management'],
            collaboration_type='project_based',
            meeting_frequency='daily',
            communication_channels=['Slack', 'Teams', 'Email'],
            collaboration_score=0.0,
            project_success_rate=0.0
        )
        
        # Trading Operations + Technology Collaboration
        self.collaborations['trading_tech'] = TeamCollaboration(
            teams=['Trading Operations', 'Technology'],
            collaboration_type='cross_functional',
            meeting_frequency='daily',
            communication_channels=['Slack', 'Jira', 'Teams'],
            collaboration_score=0.0,
            project_success_rate=0.0
        )
        
        # All Teams Collaboration (Weekly Meeting)
        self.collaborations['all_teams'] = TeamCollaboration(
            teams=['Quantitative Research', 'Data Science', 'Portfolio Management', 
                   'Risk Management', 'Trading Operations', 'Technology', 'Compliance'],
            collaboration_type='hierarchical',
            meeting_frequency='weekly',
            communication_channels=['Zoom', 'Slack', 'Email'],
            collaboration_score=0.0,
            project_success_rate=0.0
        )
        
        logger.info(f"Initialized {len(self.collaborations)} team collaborations")
    
    def add_team_member(self, employee_id: str, name: str, role_title: str) -> bool:
        """Add team member to organization"""
        try:
            role = self.roles.get(role_title)
            if not role:
                logger.error(f"Role {role_title} not found")
                return False
            
            member = TeamMember(
                employee_id=employee_id,
                name=name,
                role=role,
                next_review=datetime.utcnow() + timedelta(days=90)
            )
            
            self.members[employee_id] = member
            
            # Update metrics
            self.metrics['total_employees'] += 1
            if member.is_active:
                self.metrics['active_employees'] += 1
            
            logger.info(f"Added team member: {name} as {role_title}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add team member: {e}")
            return False
    
    def calculate_compensation(self, employee_id: str, performance_score: float, 
                             fund_performance: float) -> Dict[str, Any]:
        """Calculate total compensation for team member"""
        try:
            member = self.members.get(employee_id)
            if not member:
                return {'error': 'Employee not found'}
            
            role = member.role
            
            # Base salary (midpoint of range)
            base_salary = sum(role.base_salary_range) / 2
            
            # Performance bonus
            performance_multiplier = performance_score / 5.0  # Normalize to 0-1
            performance_bonus = base_salary * role.bonus_percentage * performance_multiplier
            
            # Fund performance bonus
            fund_multiplier = max(0, fund_performance) / 0.15  # 15% target return
            fund_bonus = base_salary * role.bonus_percentage * fund_multiplier * 0.5
            
            # Total bonus
            total_bonus = performance_bonus + fund_bonus
            
            # Equity value (simplified)
            equity_value = base_salary * role.equity_percentage * 10  # 10x multiple
            
            # Total compensation
            total_compensation = base_salary + total_bonus + equity_value
            
            return {
                'base_salary': base_salary,
                'performance_bonus': performance_bonus,
                'fund_bonus': fund_bonus,
                'total_bonus': total_bonus,
                'equity_value': equity_value,
                'total_compensation': total_compensation,
                'breakdown': {
                    'base_percentage': base_salary / total_compensation * 100,
                    'bonus_percentage': total_bonus / total_compensation * 100,
                    'equity_percentage': equity_value / total_compensation * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Compensation calculation failed: {e}")
            return {'error': str(e)}
    
    def evaluate_team_performance(self) -> Dict[str, Any]:
        """Evaluate overall team performance"""
        try:
            # Calculate average performance
            active_members = [m for m in self.members.values() if m.is_active]
            if active_members:
                avg_performance = sum(m.performance_score for m in active_members) / len(active_members)
            else:
                avg_performance = 0.0
            
            # Calculate total compensation
            total_comp = sum(
                sum(m.role.base_salary_range) / 2 * (1 + m.role.bonus_percentage + m.role.equity_percentage)
                for m in active_members
            )
            
            # Team composition analysis
            team_composition = {}
            for member in active_members:
                dept = member.role.department
                if dept not in team_composition:
                    team_composition[dept] = {'count': 0, 'avg_performance': 0.0}
                team_composition[dept]['count'] += 1
                team_composition[dept]['avg_performance'] += member.performance_score
            
            # Calculate department averages
            for dept in team_composition:
                if team_composition[dept]['count'] > 0:
                    team_composition[dept]['avg_performance'] /= team_composition[dept]['count']
            
            # Collaboration metrics
            collaboration_scores = [c.collaboration_score for c in self.collaborations.values()]
            avg_collaboration = sum(collaboration_scores) / len(collaboration_scores) if collaboration_scores else 0.0
            
            return {
                'total_employees': len(active_members),
                'average_performance': avg_performance,
                'total_compensation': total_comp,
                'team_composition': team_composition,
                'collaboration_metrics': {
                    'average_score': avg_collaboration,
                    'total_collaborations': len(self.collaborations)
                },
                'performance_distribution': {
                    'excellent': len([m for m in active_members if m.performance_score >= 4.5]),
                    'good': len([m for m in active_members if 3.5 <= m.performance_score < 4.5]),
                    'average': len([m for m in active_members if 2.5 <= m.performance_score < 3.5]),
                    'below_average': len([m for m in active_members if m.performance_score < 2.5])
                }
            }
            
        except Exception as e:
            logger.error(f"Team performance evaluation failed: {e}")
            return {'error': str(e)}
    
    def get_hiring_plan(self, target_aum: float) -> Dict[str, Any]:
        """Generate hiring plan based on target AUM"""
        try:
            # Calculate required team size based on AUM
            # Industry standard: ~1 employee per $50M AUM for small funds
            # Scale down for efficiency: 1 employee per $100M AUM
            
            required_employees = max(10, int(target_aum / 100000000))  # Minimum 10 employees
            
            current_employees = len([m for m in self.members.values() if m.is_active])
            employees_to_hire = max(0, required_employees - current_employees)
            
            # Define hiring priorities
            hiring_priorities = [
                {'role': 'senior_quant_researcher', 'priority': 'high', 'count': 2},
                {'role': 'senior_data_scientist', 'priority': 'high', 'count': 2},
                {'role': 'portfolio_manager', 'priority': 'high', 'count': 1},
                {'role': 'senior_risk_manager', 'priority': 'medium', 'count': 1},
                {'role': 'senior_software_engineer', 'priority': 'medium', 'count': 2},
                {'role': 'senior_trader', 'priority': 'medium', 'count': 1},
                {'role': 'quant_researcher', 'priority': 'low', 'count': 3},
                {'role': 'head_trading_operations', 'priority': 'low', 'count': 1}
            ]
            
            # Calculate hiring costs
            total_hiring_cost = 0
            hiring_breakdown = []
            
            for priority in hiring_priorities:
                if employees_to_hire <= 0:
                    break
                
                role = self.roles.get(priority['role'])
                if not role:
                    continue
                
                hire_count = min(priority['count'], employees_to_hire)
                avg_salary = sum(role.base_salary_range) / 2
                total_cost = avg_salary * hire_count
                
                total_hiring_cost += total_cost
                employees_to_hire -= hire_count
                
                hiring_breakdown.append({
                    'role': priority['role'],
                    'count': hire_count,
                    'priority': priority['priority'],
                    'average_salary': avg_salary,
                    'total_cost': total_cost
                })
            
            return {
                'target_aum': target_aum,
                'required_employees': required_employees,
                'current_employees': current_employees,
                'employees_to_hire': required_employees - current_employees,
                'hiring_breakdown': hiring_breakdown,
                'total_hiring_cost': total_hiring_cost,
                'estimated_timeline': f"{max(1, (required_employees - current_employees) // 2)} months",
                'recruitment_strategy': {
                    'channels': ['LinkedIn', 'Industry conferences', 'Executive search firms', 'University recruiting'],
                    'timeline': '3-6 months for senior roles, 1-2 months for junior roles',
                    'budget_allocation': {
                        'recruitment_fees': total_hiring_cost * 0.2,
                        'onboarding_costs': total_hiring_cost * 0.1,
                        'first_year_compensation': total_hiring_cost * 1.2
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Hiring plan generation failed: {e}")
            return {'error': str(e)}
    
    def get_team_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive team dashboard"""
        try:
            # Team performance evaluation
            performance = self.evaluate_team_performance()
            
            # Department breakdown
            departments = {}
            for member in self.members.values():
                if member.is_active:
                    dept = member.role.department
                    if dept not in departments:
                        departments[dept] = {
                            'count': 0,
                            'avg_performance': 0.0,
                            'avg_salary': 0.0,
                            'total_compensation': 0.0
                        }
                    
                    departments[dept]['count'] += 1
                    departments[dept]['avg_performance'] += member.performance_score
                    
                    avg_salary = sum(member.role.base_salary_range) / 2
                    departments[dept]['avg_salary'] += avg_salary
                    departments[dept]['total_compensation'] += avg_salary * (1 + member.role.bonus_percentage + member.role.equity_percentage)
            
            # Calculate averages
            for dept in departments:
                if departments[dept]['count'] > 0:
                    departments[dept]['avg_performance'] /= departments[dept]['count']
                    departments[dept]['avg_salary'] /= departments[dept]['count']
            
            # Role distribution
            role_distribution = {}
            for member in self.members.values():
                if member.is_active:
                    role_title = member.role.title
                    if role_title not in role_distribution:
                        role_distribution[role_title] = 0
                    role_distribution[role_title] += 1
            
            # Collaboration metrics
            collaboration_metrics = {}
            for name, collab in self.collaborations.items():
                collaboration_metrics[name] = {
                    'teams': collab.teams,
                    'meeting_frequency': collab.meeting_frequency,
                    'collaboration_score': collab.collaboration_score,
                    'project_success_rate': collab.project_success_rate
                }
            
            return {
                'team_overview': {
                    'total_employees': len([m for m in self.members.values() if m.is_active]),
                    'average_performance': performance.get('average_performance', 0.0),
                    'total_compensation': performance.get('total_compensation', 0.0),
                    'departments': len(departments),
                    'roles': len(role_distribution)
                },
                'departments': departments,
                'role_distribution': role_distribution,
                'collaboration_metrics': collaboration_metrics,
                'performance_metrics': performance,
                'hiring_needs': self.get_hiring_plan(100000000)  # $100M AUM example
            }
            
        except Exception as e:
            logger.error(f"Team dashboard generation failed: {e}")
            return {'error': str(e)}


# Global team structure instance
_team_structure = None

def get_team_structure() -> TeamStructure:
    """Get global team structure instance"""
    global _team_structure
    if _team_structure is None:
        _team_structure = TeamStructure()
    return _team_structure


if __name__ == "__main__":
    # Test team structure
    team = TeamStructure()
    
    # Add sample team members
    team.add_team_member('E001', 'Dr. Alice Chen', 'head_quant_research')
    team.add_team_member('E002', 'Dr. Bob Smith', 'senior_data_scientist')
    team.add_team_member('E003', 'Carol Johnson', 'portfolio_manager')
    team.add_team_member('E004', 'David Lee', 'senior_risk_manager')
    team.add_team_member('E005', 'Eve Wilson', 'cto')
    
    # Calculate compensation
    comp = team.calculate_compensation('E001', 4.5, 0.18)
    print(f"Compensation for E001: {comp}")
    
    # Evaluate team performance
    performance = team.evaluate_team_performance()
    print(f"Team performance: {json.dumps(performance, indent=2, default=str)}")
    
    # Get hiring plan
    hiring = team.get_hiring_plan(500000000)  # $500M AUM
    print(f"Hiring plan: {json.dumps(hiring, indent=2, default=str)}")
    
    # Get team dashboard
    dashboard = team.get_team_dashboard()
    print(f"Team dashboard: {json.dumps(dashboard, indent=2, default=str)}")
