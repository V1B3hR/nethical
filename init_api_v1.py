#!/usr/bin/env python3
"""
Initialize Nethical API v1 database and create default admin user.

This script sets up the database and creates a default admin user
for first-time setup.

Usage:
    python init_api_v1.py

Options:
    --username USERNAME    Admin username (default: admin)
    --password PASSWORD    Admin password (default: admin123)
    --email EMAIL         Admin email (default: admin@example.com)
"""

import argparse
import sys

from nethical.database import SessionLocal, User, init_db
from nethical.api.rbac import get_password_hash


def create_admin_user(username: str, password: str, email: str) -> bool:
    """Create default admin user.
    
    Args:
        username: Admin username
        password: Admin password
        email: Admin email
        
    Returns:
        True if successful, False otherwise
    """
    db = SessionLocal()
    
    try:
        # Check if user already exists
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            print(f"✗ User '{username}' already exists")
            return False
        
        # Create admin user
        admin = User(
            username=username,
            email=email,
            full_name="Admin User",
            hashed_password=get_password_hash(password),
            role="admin",
            is_active=True,
        )
        
        db.add(admin)
        db.commit()
        db.refresh(admin)
        
        print(f"✓ Created admin user: {username}")
        print(f"  Email: {email}")
        print(f"  Role: {admin.role}")
        print(f"  ID: {admin.id}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error creating admin user: {e}")
        db.rollback()
        return False
    
    finally:
        db.close()


def create_sample_agent(db_session) -> bool:
    """Create a sample agent for demonstration.
    
    Args:
        db_session: Database session
        
    Returns:
        True if successful
    """
    from nethical.database import Agent
    
    try:
        # Check if sample agent exists
        existing = db_session.query(Agent).filter(Agent.agent_id == "sample-gpt4").first()
        if existing:
            print("✓ Sample agent already exists")
            return True
        
        # Create sample agent
        agent = Agent(
            agent_id="sample-gpt4",
            name="Sample GPT-4 Agent",
            agent_type="llm",
            description="Sample agent for testing",
            trust_level=0.8,
            status="active",
            configuration={"model": "gpt-4", "temperature": 0.7},
            meta_data={"environment": "development"},
            created_by="admin",
        )
        
        db_session.add(agent)
        db_session.commit()
        
        print("✓ Created sample agent: sample-gpt4")
        return True
    
    except Exception as e:
        print(f"✗ Error creating sample agent: {e}")
        db_session.rollback()
        return False


def create_sample_policy(db_session) -> bool:
    """Create a sample policy for demonstration.
    
    Args:
        db_session: Database session
        
    Returns:
        True if successful
    """
    from nethical.database import Policy
    
    try:
        # Check if sample policy exists
        existing = db_session.query(Policy).filter(Policy.policy_id == "sample-policy").first()
        if existing:
            print("✓ Sample policy already exists")
            return True
        
        # Create sample policy
        policy = Policy(
            policy_id="sample-policy",
            name="Sample Security Policy",
            description="Sample policy for demonstration",
            version="1.0.0",
            policy_type="governance",
            priority=100,
            status="active",
            rules=[
                {
                    "id": "rule-1",
                    "condition": "risk_score > 0.8",
                    "action": "BLOCK",
                    "priority": 10,
                    "description": "Block high-risk actions"
                }
            ],
            scope="global",
            fundamental_laws=[1, 2, 3],
            meta_data={"category": "security"},
            created_by="admin",
        )
        
        db_session.add(policy)
        db_session.commit()
        
        print("✓ Created sample policy: sample-policy")
        return True
    
    except Exception as e:
        print(f"✗ Error creating sample policy: {e}")
        db_session.rollback()
        return False


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description="Initialize Nethical API v1 database")
    parser.add_argument("--username", default="admin", help="Admin username")
    parser.add_argument("--password", default="admin123", help="Admin password")
    parser.add_argument("--email", default="admin@example.com", help="Admin email")
    parser.add_argument("--no-samples", action="store_true", help="Don't create sample data")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Nethical API v1 Initialization")
    print("=" * 60)
    
    try:
        # Initialize database
        print("\nInitializing database...")
        init_db()
        print("✓ Database initialized")
        
        # Create admin user
        print("\nCreating admin user...")
        if not create_admin_user(args.username, args.password, args.email):
            # User already exists, but that's ok
            pass
        
        # Create sample data
        if not args.no_samples:
            print("\nCreating sample data...")
            db = SessionLocal()
            try:
                create_sample_agent(db)
                create_sample_policy(db)
            finally:
                db.close()
        
        # Success
        print("\n" + "=" * 60)
        print("  Initialization Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Start the server:")
        print("     uvicorn nethical.api.v1.app:create_v1_app --factory --reload")
        print("\n  2. Login credentials:")
        print(f"     Username: {args.username}")
        print(f"     Password: {args.password}")
        print("\n  3. Access documentation:")
        print("     http://localhost:8000/docs")
        print("\n  4. Run demo:")
        print("     python demo_api_v1.py")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
