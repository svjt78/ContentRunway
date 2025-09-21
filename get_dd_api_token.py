#!/usr/bin/env python3
"""Get DigitalDossier API token for testing."""

import requests
import os

def get_api_token():
    """Get API token from DigitalDossier."""
    
    base_url = os.getenv('DIGITALDOSSIER_BASE_URL', 'http://localhost:3003')
    admin_email = os.getenv('DIGITALDOSSIER_ADMIN_EMAIL')
    admin_password = os.getenv('DIGITALDOSSIER_ADMIN_PASSWORD')
    
    print("🔑 Getting DigitalDossier API Token...")
    print(f"🌐 Base URL: {base_url}")
    
    if not admin_email or not admin_password:
        print("❌ Admin credentials not set")
        print("Please set DIGITALDOSSIER_ADMIN_EMAIL and DIGITALDOSSIER_ADMIN_PASSWORD")
        print("\nAlternatively, you can:")
        print("1. Go to http://localhost:3003/admin")
        print("2. Log in to the admin panel") 
        print("3. Navigate to API settings")
        print("4. Generate or copy your API token")
        print("5. Set DIGITALDOSSIER_API_TOKEN environment variable")
        return None
    
    try:
        # Try to get API token via admin login
        login_url = f"{base_url}/admin/api/login"
        
        login_data = {
            "email": admin_email,
            "password": admin_password
        }
        
        print(f"🔐 Attempting login to {login_url}...")
        
        response = requests.post(login_url, json=login_data, timeout=10)
        
        if response.status_code == 200:
            token_data = response.json()
            api_token = token_data.get('token') or token_data.get('api_token')
            
            if api_token:
                print("✅ API Token retrieved successfully!")
                print(f"🔑 Token: {api_token[:20]}...")
                print(f"\n📝 Add this to your .env file:")
                print(f"DIGITALDOSSIER_API_TOKEN={api_token}")
                return api_token
            else:
                print("❌ No token in response")
                print(f"Response: {token_data}")
        else:
            print(f"❌ Login failed: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error getting token: {e}")
    
    print("\n💡 Manual token retrieval:")
    print("1. Open http://localhost:3003/admin in your browser")
    print("2. Log in with your admin credentials")
    print("3. Go to Settings > API Keys")
    print("4. Generate or copy an API token")
    print("5. Set: export DIGITALDOSSIER_API_TOKEN=your_token_here")
    
    return None

if __name__ == "__main__":
    # You can set these temporarily for testing
    # os.environ['DIGITALDOSSIER_ADMIN_EMAIL'] = 'admin@example.com'  
    # os.environ['DIGITALDOSSIER_ADMIN_PASSWORD'] = 'your_password'
    
    token = get_api_token()
    
    if token:
        print(f"\n🚀 Ready to test uploads!")
        print(f"Run: export DIGITALDOSSIER_API_TOKEN={token}")
        print(f"Then run the upload test script")
    else:
        print(f"\n⚠️ Please manually set up the API token")