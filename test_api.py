"""
Face Recognition API - Test Script
==================================
Simple test script to verify the API is working correctly.
Run this after starting the Flask server.
"""

import requests
import os
import sys

BASE_URL = "http://localhost:5000"


def test_health():
    """Test the health check endpoint."""
    print("\n🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        if response.status_code == 200:
            print(f"   ✅ API Status: {data.get('api')}")
            print(f"   ✅ Database Status: {data.get('database')}")
            return True
        else:
            print(f"   ❌ Health check failed: {data}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server. Is it running?")
        return False


def test_home():
    """Test the home endpoint."""
    print("\n🏠 Testing Home Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        data = response.json()
        
        print(f"   Service: {data.get('service')}")
        print(f"   Version: {data.get('version')}")
        print(f"   Endpoints available: {len(data.get('endpoints', {}))}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_list_users():
    """Test listing all users."""
    print("\n👥 Testing List Users...")
    try:
        response = requests.get(f"{BASE_URL}/users")
        data = response.json()
        
        print(f"   Total registered users: {data.get('count', 0)}")
        users = data.get('users', [])
        for user in users[:5]:  # Show first 5 users
            print(f"   - {user.get('user_id')}: {user.get('name', 'No name')}")
        
        if len(users) > 5:
            print(f"   ... and {len(users) - 5} more")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_register(image_path, user_id, name="Test User"):
    """Test registering a face."""
    print(f"\n📝 Testing Face Registration...")
    
    if not os.path.exists(image_path):
        print(f"   ❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'user_id': user_id, 'name': name}
            
            response = requests.post(f"{BASE_URL}/register", files=files, data=data)
            result = response.json()
            
            if result.get('status') == 'success':
                print(f"   ✅ {result.get('message')}")
                print(f"   User ID: {result.get('user_id')}")
                return True
            else:
                print(f"   ❌ Registration failed: {result.get('error')}")
                return False
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_verify(image_path):
    """Test verifying a face."""
    print(f"\n🔐 Testing Face Verification...")
    
    if not os.path.exists(image_path):
        print(f"   ❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': img_file}
            data = {'tolerance': 0.5}
            
            response = requests.post(f"{BASE_URL}/verify", files=files, data=data)
            result = response.json()
            
            if result.get('matched'):
                print(f"   ✅ Face matched!")
                print(f"   User ID: {result.get('user_id')}")
                print(f"   Name: {result.get('name')}")
                print(f"   Confidence: {result.get('confidence')}")
                return True
            else:
                print(f"   ⚠️ No match found")
                print(f"   Best distance: {result.get('best_distance')}")
                return False
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_delete(user_id):
    """Test deleting a user."""
    print(f"\n🗑️ Testing Delete User...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/delete",
            json={'user_id': user_id}
        )
        result = response.json()
        
        if result.get('status') == 'success':
            print(f"   ✅ {result.get('message')}")
            return True
        else:
            print(f"   ⚠️ {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    print("=" * 50)
    print("🎭 Face Recognition API - Test Suite")
    print("=" * 50)
    
    # Basic connectivity tests
    if not test_health():
        print("\n⚠️ Server not responding. Please start the server first:")
        print("   python app.py")
        sys.exit(1)
    
    test_home()
    test_list_users()
    
    # Image-based tests (only if image provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        user_id = sys.argv[2] if len(sys.argv) > 2 else "test_user_001"
        name = sys.argv[3] if len(sys.argv) > 3 else "Test User"
        
        print("\n" + "-" * 50)
        print("Running image-based tests...")
        print("-" * 50)
        
        # Register
        if test_register(image_path, user_id, name):
            # Verify the same image
            test_verify(image_path)
            
            # Optionally delete
            confirm = input("\n   Delete test user? (y/n): ").strip().lower()
            if confirm == 'y':
                test_delete(user_id)
    else:
        print("\n" + "-" * 50)
        print("To run full tests with image registration/verification:")
        print("   python test_api.py <image_path> [user_id] [name]")
        print("\nExample:")
        print("   python test_api.py ./myface.jpg user123 'John Doe'")
        print("-" * 50)
    
    print("\n✨ Test suite completed!")


if __name__ == "__main__":
    main()
