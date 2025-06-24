"""
TelusHolding API Client for inserting leads
"""
import requests
import logging
from typing import Dict, Any, Tuple
from config import TELUS_CONFIG

logger = logging.getLogger(__name__)

class TelusAPIClient:
    """Client for interacting with TelusHolding API"""
    
    def __init__(self):
        self.config = TELUS_CONFIG
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'TelusHolding-CrossSell-Client/1.0'
        })
    
    def _map_client_to_lead(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map client data to TelusHolding lead format
        Adjust this mapping based on your actual CSV structure
        """
        # Default mapping - adjust based on your CSV columns
        lead_data = {
            'PassKey': self.config.API_KEY,
            'puissanceFiscale': client_data.get('puissance_fiscale', ''),
            'energie': client_data.get('energie', 'Essence'),  # Default to Essence
            'sexe': self._map_gender(client_data.get('gender', client_data.get('sexe', ''))),
            'telephone': client_data.get('phone', client_data.get('telephone', '')),
            'nom': client_data.get('last_name', client_data.get('nom', '')),
            'prenom': client_data.get('first_name', client_data.get('prenom', '')),
            'ville': client_data.get('city', client_data.get('ville', '')),
            'agent': client_data.get('agent', 'CROSS_SELL_AUTO'),
            'codePromo': client_data.get('promo_code', client_data.get('codePromo', 'AUTO_CROSS_SELL')),
            'categorie': client_data.get('category', client_data.get('categorie', 'AUTOMOBILE')),
            'cin': client_data.get('cin', ''),
            'conventionLibelle': client_data.get('convention_libelle', client_data.get('conventionLibelle', '')),
            'dateEcheanceSyn': client_data.get('date_echeance_syn', client_data.get('dateEcheanceSyn', ''))
        }
        
        return lead_data
    
    def _map_gender(self, gender: str) -> str:
        """Map gender to the format expected by TelusHolding"""
        if not gender:
            return ''
        
        gender_lower = gender.lower()
        if gender_lower in ['m', 'male', 'homme', 'masculin']:
            return 'M'
        elif gender_lower in ['f', 'female', 'femme', 'feminin']:
            return 'F'
        else:
            return gender.upper()
    
    def insert_lead(self, client_data: Dict[str, Any]) -> Tuple[bool, str, Any]:
        """
        Insert a lead into TelusHolding CRM
        
        Returns:
            Tuple[bool, str, Any]: (success, message, response_data)
        """
        try:
            # Map client data to lead format
            lead_data = self._map_client_to_lead(client_data)
            
            # Validate required fields
            required_fields = ['telephone', 'nom', 'prenom']
            missing_fields = [field for field in required_fields if not lead_data.get(field)]
            
            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                logger.error(error_msg)
                return False, error_msg, None
            
            # Make API request
            url = self.config.insert_url
            logger.info(f"Sending lead to TelusHolding API: {url}")
            logger.debug(f"Lead data: {lead_data}")
            
            response = self.session.post(url, data=lead_data, timeout=30)
            response.raise_for_status()
            
            # Parse response
            response_text = response.text.strip()
            success, message = self._parse_response(response_text)
            
            if success:
                logger.info(f"Lead inserted successfully: {message}")
            else:
                logger.error(f"Lead insertion failed: {message}")
            
            return success, message, response_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Unexpected error inserting lead: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def _parse_response(self, response_text: str) -> Tuple[bool, str]:
        """
        Parse TelusHolding API response
        
        Response codes:
        0: PassKey manquant
        1: PassKey incorrect
        2: Donné important manquant
        3: Insertion en double
        4: Erreur d'insertion
        5 + ID LEAD: Insertion réussi
        """
        response_messages = {
            '0': 'PassKey manquant',
            '1': 'PassKey incorrect',
            '2': 'Données importantes manquantes',
            '3': 'Insertion en double',
            '4': 'Erreur d\'insertion'
        }
        
        if response_text.startswith('5'):
            # Successful insertion
            lead_id = response_text.replace('5', '').strip()
            return True, f"Lead inserted successfully with ID: {lead_id}"
        elif response_text in response_messages:
            return False, response_messages[response_text]
        else:
            return False, f"Unknown response: {response_text}"
    
    def test_connection(self) -> bool:
        """Test the API connection"""
        try:
            # Create a minimal test payload
            test_data = {
                'PassKey': self.config.API_KEY,
                'telephone': '0600000000',
                'nom': 'TEST',
                'prenom': 'CONNECTION',
                'ville': 'TEST',
                'agent': 'TEST',
                'codePromo': 'TEST',
                'categorie': 'TEST'
            }
            
            response = self.session.post(
                self.config.insert_url, 
                data=test_data, 
                timeout=10
            )
            
            # Even if the test data is rejected, a proper response means connection works
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False