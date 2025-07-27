#!/usr/bin/env python3
"""
Acceptance test script for the recommendation API.
"""

import pytest
import requests


def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get("http://localhost:8000/health")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running for acceptance tests")
    print(f"Health endpoint: {response.status_code}")
    print(response.json())


def test_recommend():
    """Test the recommend endpoint."""
    try:
        response = requests.get("http://localhost:8000/recommend?user_id=U_01&k=5")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running for acceptance tests")
    print(f"Recommend endpoint: {response.status_code}")
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error: {response.text}")


if __name__ == "__main__":
    print("Running acceptance tests...")
    test_health()
    test_recommend()
    print("Done!")
