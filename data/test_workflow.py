import requests
import os

# This script is designed to run inside the Docker container
BASE_URL = "http://localhost:5000"
TEST_USER = "musk_001"
TEST_NAME = "Elon Musk"
TOKEN = "staging-token-12345"
HEADERS = {"Authorization": "Bearer {0}".format(TOKEN)}

def run_test():
    print("="*50)
    print("🚀 STARTING FAMOUS PERSON TEST WORKFLOW")
    print("="*50)

    # 1. Register with Image 1
    print("\n1️⃣ Registering {0}...".format(TEST_NAME))
    img1_path = "/app/data/test_data/elon_1.jpg"
    if not os.path.exists(img1_path):
        # Fallback for local path if running outside docker
        img1_path = "data/test_data/elon_1.jpg"
        
    if not os.path.exists(img1_path):
        print("❌ Error: Image not found.")
        return

    try:
        with open(img1_path, 'rb') as img:
            files = {'image': img}
            data = {'user_id': TEST_USER, 'name': TEST_NAME}
            resp = requests.post("{0}/register".format(BASE_URL), files=files, data=data, headers=HEADERS)
            print("Response Status: {0}".format(resp.status_code))
            print("Response Data: {0}".format(resp.json()))
    except Exception as e:
        print("❌ Register Error: {0}".format(e))
        return

    # 2. Verify with same Image 1
    print("\n2️⃣ Verifying with SAME image...")
    with open(img1_path, 'rb') as img:
        files = {'image': img}
        data = {'user_id': TEST_USER}
        resp = requests.post("{0}/verify".format(BASE_URL), files=files, data=data, headers=HEADERS)
        res = resp.json()
        print("Result: Match={0}, Confidence={1}%".format(res.get('matched'), res.get('confidence')))

    # 3. Verify with DIFFERENT Image 2
    print("\n3️⃣ Verifying with DIFFERENT image...")
    img2_path = "/app/data/test_data/elon_2.jpg"
    if not os.path.exists(img2_path):
        img2_path = "data/test_data/elon_2.jpg"
        
    if os.path.exists(img2_path):
        with open(img2_path, 'rb') as img:
            files = {'image': img}
            data = {'user_id': TEST_USER}
            resp = requests.post("{0}/verify".format(BASE_URL), files=files, data=data, headers=HEADERS)
            res = resp.json()
            print("Result: Match={0}, Confidence={1}%".format(res.get('matched'), res.get('confidence')))

    # 4. List users
    print("\n4️⃣ Checking User List...")
    resp = requests.get("{0}/users".format(BASE_URL), headers=HEADERS)
    users = resp.json().get('users', [])
    found = any(u['user_id'] == TEST_USER for u in users)
    print("User in list: {0}".format('✅ Found' if found else '❌ Missing'))

    print("\n" + "="*50)
    print("✨ TEST WORKFLOW FINISHED")
    print("="*50)

if __name__ == "__main__":
    run_test()
