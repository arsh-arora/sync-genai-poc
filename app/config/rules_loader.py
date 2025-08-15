"""
Centralized rules loader for synchrony-demo-rules-repo
Loads all YAML rules files at application startup
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class RulesLoader:
    """Centralized loader for all rules from synchrony-demo-rules-repo"""
    
    def __init__(self, rules_base_path: str = "synchrony-demo-rules-repo"):
        self.rules_base = Path(rules_base_path)
        self.rules = {}
        self.fixtures = {}
        self._load_all_rules()
        self._load_all_fixtures()
    
    def _load_all_rules(self):
        """Load all YAML rules files"""
        rules_dir = self.rules_base / "rules"
        
        if not rules_dir.exists():
            logger.warning(f"Rules directory not found: {rules_dir}")
            return
        
        rule_files = [
            "promotions.yml",
            "disclosures.yml", 
            "prequalification.yml",
            "trustshield.yml",
            "collections.yml",
            "contracts_lexicon.yml",
            "dispute.yml",
            "carecredit.yml",
            "narrator.yml",
            "imagegen.yml",
            "routing.yml"
        ]
        
        loaded_count = 0
        for rule_file in rule_files:
            rule_path = rules_dir / rule_file
            if rule_path.exists():
                try:
                    with open(rule_path, 'r') as f:
                        rule_name = rule_file.replace('.yml', '')
                        self.rules[rule_name] = yaml.safe_load(f)
                        loaded_count += 1
                        logger.debug(f"Loaded rules: {rule_file}")
                except Exception as e:
                    logger.error(f"Failed to load {rule_file}: {e}")
            else:
                logger.warning(f"Rule file not found: {rule_file}")
        
        logger.info(f"Loaded {loaded_count} rules files from synchrony-demo-rules-repo")
    
    def _load_all_fixtures(self):
        """Load all fixture JSON files"""
        fixtures_dir = self.rules_base / "fixtures"
        
        if not fixtures_dir.exists():
            logger.warning(f"Fixtures directory not found: {fixtures_dir}")
            return
        
        fixture_files = [
            "products.json",
            "merchants.json"
        ]
        
        loaded_count = 0
        for fixture_file in fixture_files:
            fixture_path = fixtures_dir / fixture_file
            if fixture_path.exists():
                try:
                    with open(fixture_path, 'r') as f:
                        fixture_name = fixture_file.replace('.json', '')
                        self.fixtures[fixture_name] = json.load(f)
                        loaded_count += 1
                        logger.debug(f"Loaded fixture: {fixture_file}")
                except Exception as e:
                    logger.error(f"Failed to load {fixture_file}: {e}")
            else:
                logger.warning(f"Fixture file not found: {fixture_file}")
        
        logger.info(f"Loaded {loaded_count} fixture files from synchrony-demo-rules-repo")
    
    def get_rules(self, rule_type: str) -> Optional[Dict[str, Any]]:
        """Get rules by type (e.g., 'promotions', 'trustshield')"""
        return self.rules.get(rule_type)
    
    def get_fixture(self, fixture_type: str) -> Optional[Dict[str, Any]]:
        """Get fixture by type (e.g., 'products', 'merchants')"""
        return self.fixtures.get(fixture_type)
    
    def get_all_rules(self) -> Dict[str, Any]:
        """Get all loaded rules"""
        return self.rules.copy()
    
    def get_all_fixtures(self) -> Dict[str, Any]:
        """Get all loaded fixtures"""
        return self.fixtures.copy()
    
    def reload_rules(self):
        """Reload all rules from files"""
        self.rules = {}
        self.fixtures = {}
        self._load_all_rules()
        self._load_all_fixtures()

# Global rules loader instance
_rules_loader: Optional[RulesLoader] = None

def get_rules_loader() -> RulesLoader:
    """Get the global rules loader instance"""
    global _rules_loader
    if _rules_loader is None:
        _rules_loader = RulesLoader()
    return _rules_loader

def get_rules(rule_type: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get rules by type"""
    return get_rules_loader().get_rules(rule_type)

def get_fixture(fixture_type: str) -> Optional[Dict[str, Any]]:
    """Convenience function to get fixture by type"""
    return get_rules_loader().get_fixture(fixture_type)