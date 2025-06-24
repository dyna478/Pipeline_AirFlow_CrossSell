#!/usr/bin/env python
"""
Test script to verify TelusHolding API connection using POST requests
"""
import os
import sys
import json

# Set default test API key
if not os.environ.get('TELUS_API_KEY'):
    os.environ['TELUS_API_KEY'] = 'test_api_key'  # Default for testing - replace with real key

# Always use test environment for this test script
os.environ['TELUS_USE_TEST'] = 'true'

# Add cross_sell_auto to path if it exists
if os.path.exists("./cross_sell_auto"):
    sys.path.append("./cross_sell_auto")
    print("Added cross_sell_auto to path")

print("===== TESTING TELUSHOLDING API CONNECTION =====")

try:
    # Import the TelusAPIClient
    print("Importing TelusAPIClient...")
    from cross_sell_auto.telus_api_client import TelusAPIClient
    
    print("\nInitializing API client...")
    api_client = TelusAPIClient()
    
    print(f"\nTest Environment Configured:")
    print(f"- API Key: {os.environ.get('TELUS_API_KEY')[:5]}...")  # Show just first 5 chars for security
    print(f"- Test mode: {os.environ.get('TELUS_USE_TEST')}")
    
    # Test API connection
    print("\nTesting API connection...")
    success = api_client.test_connection()
    if success:
        print("✓ API connection successful")
    else:
        print("✗ API connection failed")
    
    # Test sending a lead
    print("\nTesting lead creation with POST...")
    test_lead = {
        "firstname": "Test",
        "lastname": "API",
        "emailaddress": "test@example.com",
        "phonenumber": "5551234567",
        "city": "Montreal",
        "gender": "M",
        "agent": "test_script",
        "promo_code": "TESTAUTO",
        "category": "AUTO",
        "power": "150",
        "energy": "essence",
        "notes": "This is a test lead from API test script"
    }
    
    lead_response = api_client.send_lead(test_lead)
    print(f"API Response: {json.dumps(lead_response, indent=2)}")
    
    if lead_response.get("success"):
        print(f"✓ Lead creation successful!")
        
        # Test updating the lead if we got a lead_id
        lead_id = lead_response.get("lead_id") or "12345"  # Use 12345 as fallback for testing
        print(f"\nTesting lead update with POST...")
        update_response = api_client.update_lead(
            lead_id=lead_id,
            policy_number="TEST12345",
            status="CONTACTED"
        )
        print(f"Update Response: {json.dumps(update_response, indent=2)}")
        
        if update_response.get("success"):
            print(f"✓ Lead update successful!")
        else:
            print(f"✗ Lead update failed: {update_response.get('message')}")
    else:
        print(f"✗ Lead creation failed: {lead_response.get('message')}")
    
    print("\n===== TEST COMPLETED =====")
    
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 