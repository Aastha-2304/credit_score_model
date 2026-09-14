import requests
from fastapi.testclient import TestClient
from api.main import app

def test_api():
    client = TestClient(app)
    
    # 1. Health check
    res = client.get('/health')
    print('Health check:', res.json())
    assert res.status_code == 200

    # 2. Score applicant
    applicant_payload = {
        'loan_amnt': 12000.0,
        'term': 36.0,
        'int_rate': 10.5,
        'grade': 'B',
        'sub_grade': 'B2',
        'emp_length': '5 years',
        'home_ownership': 'MORTGAGE',
        'annual_inc': 82000.0,
        'purpose': 'debt_consolidation',
        'dti': 14.5,
        'delinq_2yrs': 0,
        'inq_last_6mths': 0,
        'pub_rec': 0,
        'revol_util': 35.0,
        'total_acc': 18
    }

    score_res = client.post('/score', json=applicant_payload)
    print('Score response status:', score_res.status_code)
    data = score_res.json()
    print('Assigned Credit Score:', data.get('credit_score'))
    print('PD:', data.get('probability_of_default'))
    print('Risk Band:', data.get('risk_band'))
    print('Recommendation:', data.get('recommendation'))
    print('Challenger PD:', data.get('challenger_pd'))
    assert score_res.status_code == 200
    assert 300 <= data.get('credit_score') <= 850
    print('ALL API TESTS PASSED!')

if __name__ == '__main__':
    test_api()
