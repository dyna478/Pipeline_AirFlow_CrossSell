    """
    Client for TelusHolding API
    """
    import os
    import json
    from datetime import datetime
    import requests
    import pickle

    class TelusAPIClient:
        def __init__(self):
            """Initialize API client with credentials from environment"""
            self.api_key = os.environ.get('TELUS_API_KEY')
            if not self.api_key:
                raise ValueError("TELUS_API_KEY environment variable not set")
                
            # Determine which environment to use
            self.use_test = os.environ.get('TELUS_USE_TEST', 'true').lower() == 'true'
            
            # Set correct API URLs 
            if self.use_test:
                # Test/sandbox environment
                self.base_url = 'https://telusholding.cloud/LEADSMANAGER/API'
                self.leads_endpoint = '/lead_insertV2_SAND.php'
                self.update_endpoint = '/lead_update_SAND.php'
            else:
                # Production environment
                self.base_url = 'http://192.168.144.10/API'
                self.leads_endpoint = '/lead_insertV2.php'
                self.update_endpoint = '/lead_update.php'
            
            print(f"TelusAPIClient initialized. Using {'TEST' if self.use_test else 'PRODUCTION'} environment")
            print(f"Base URL: {self.base_url}")
            print(f"Leads endpoint: {self.leads_endpoint}")
            
        def _get_headers(self):
            """Get headers for API requests"""
            return {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
        def test_connection(self):
            """Test API connection with real request"""
            print("Testing API connection...")
            
            try:
                # For testing, we'll try to send a minimal POST request
                url = f"{self.base_url}{self.leads_endpoint}"
                headers = self._get_headers()
                
                # Create a minimal POST request with just authentication
                data = {"api_key": self.api_key}
                
                print(f"Sending POST request to: {url}")
                response = requests.post(url, headers=headers, json=data, timeout=10)
                
                print(f"API response status: {response.status_code}")
                if hasattr(response, 'text'):
                    print(f"Response text: {response.text}")
                
                if response.status_code < 400:
                    print("API connection successful")
                    return True
                else:
                    print(f"API connection failed (Status: {response.status_code})")
                    return False
                    
            except requests.exceptions.RequestException as e:
                print(f"API connection error: {str(e)}")
                return False
            except Exception as e:
                print(f"API test failed: {str(e)}")
                return False
        
        def send_lead(self, lead_data):
            """
            Send a lead to the API using POST
            
            Args:
                lead_data (dict): The lead data to send
                
            Returns:
                dict: API response
            """
            print(f"Sending lead to API...")
            
            try:
                url = f"{self.base_url}{self.leads_endpoint}"
                headers = self._get_headers()
                
                # Prepare the data to include API key
                post_data = lead_data.copy()
                post_data["api_key"] = self.api_key
                
                print(f"Sending lead to: {url}")
                print(f"Lead data: {json.dumps(post_data, indent=2)}")
                
                # Use POST method
                response = requests.post(url, headers=headers, json=post_data, timeout=10)
                
                print(f"API response status: {response.status_code}")
                if hasattr(response, 'text'):
                    print(f"Response text: {response.text}")
                
                if response.status_code < 400:  # Any non-error status code
                    print(f"API call completed with status: {response.status_code}")
                    
                    # Try to parse JSON response if possible
                    try:
                        return response.json()
                    except ValueError:
                        # If not JSON, return the raw response with success flag
                        return {
                            "success": True,
                            "message": response.text if hasattr(response, 'text') else "No response text",
                            "raw_response": response.text if hasattr(response, 'text') else None
                        }
                else:
                    error_msg = f"API Error (Status: {response.status_code}): {response.text if hasattr(response, 'text') else 'No response text'}"
                    print(f"API call failed: {error_msg}")
                    return {
                        "success": False,
                        "message": error_msg
                    }
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Request Error: {str(e)}"
                print(f"API call failed: {error_msg}")
                return {
                    "success": False,
                    "message": error_msg
                }
        
        def update_lead(self, lead_id, policy_number, status=None):
            """
            Update a lead status in the API using POST
            
            Args:
                lead_id (str): The lead ID to update
                policy_number (str): The policy number
                status (str, optional): The status to set
                
            Returns:
                dict: API response
            """
            print(f"Updating lead: {lead_id} with policy {policy_number}")
            
            try:
                url = f"{self.base_url}{self.update_endpoint}"
                headers = self._get_headers()
                
                # Prepare update data
                update_data = {
                    "lead_id": lead_id,
                    "policy_number": policy_number,
                    "api_key": self.api_key
                }
                
                # Add status if provided
                if status:
                    update_data["status"] = status
                
                print(f"Sending update to: {url}")
                print(f"Update data: {json.dumps(update_data, indent=2)}")
                
                # Use POST method
                response = requests.post(url, headers=headers, json=update_data, timeout=10)
                
                print(f"API response status: {response.status_code}")
                if hasattr(response, 'text'):
                    print(f"Response text: {response.text}")
                
                if response.status_code < 400:
                    # Try to parse JSON response if possible
                    try:
                        return response.json()
                    except ValueError:
                        # If not JSON, return the raw response with success flag
                        return {
                            "success": True,
                            "message": response.text if hasattr(response, 'text') else "No response text",
                            "raw_response": response.text if hasattr(response, 'text') else None
                        }
                else:
                    error_msg = f"API Error (Status: {response.status_code}): {response.text if hasattr(response, 'text') else 'No response text'}"
                    print(f"Update failed: {error_msg}")
                    return {
                        "success": False,
                        "message": error_msg
                    }
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Request Error: {str(e)}"
                print(f"Update failed: {error_msg}")
                return {
                    "success": False,
                    "message": error_msg
                }
            except Exception as e:
                error_msg = f"Error updating lead: {str(e)}"
                print(error_msg)
                return {
                    "success": False,
                    "message": error_msg
                }
        
        def get_lead_status(self, lead_id):
            """
            Get lead status from API
            Note: This may need to be updated based on the actual API endpoint for status
            """
            print(f"Getting status for lead: {lead_id}")
            
            # Since we don't have a specific status endpoint provided, 
            # this is just a placeholder implementation
            return {
                "lead_id": lead_id,
                "status": "UNKNOWN",
                "note": "Status endpoint not implemented",
                "last_updated": datetime.now().isoformat()
            }

    # Chargement du modèle
    with open("models/votre_modele.pkl", "rb") as f:
        model = pickle.load(f)

    # Préparation des données pour la prédiction
    # (transformer vos données dans le format attendu par le modèle)

    # Utilisation du modèle pour les prédictions
    predictions = model.predict(donnees_client)

    # Traiter les résultats
    for client_id, prediction in zip(ids_clients, predictions):
        print(f"Client {client_id}: Probabilité de conversion {prediction:.2f}") 