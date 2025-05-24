import os
import hmac
import hashlib
from flask import Flask, request, jsonify
from linear_service import LinearService

app = Flask(__name__)

WEBHOOK_SECRET = os.getenv('LINEAR_WEBHOOK_SECRET', '')
LINEAR_API_KEY = os.getenv('LINEAR_API_KEY', '')

linear_service = LinearService(LINEAR_API_KEY)

def verify_webhook_signature():
    """Middleware function to verify webhook signature"""
    raw_body = request.get_data()
    signature = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        raw_body,
        hashlib.sha256
    ).hexdigest()
    
    print('signature', signature)
    print('request.headers', dict(request.headers))
    print('WEBHOOK_SECRET', WEBHOOK_SECRET)
    
    if signature != request.headers.get('linear-signature'):
        print('Invalid signature')
        return jsonify({'error': 'Invalid signature'}), 400
    
    return None

@app.route('/api/linear/watch-issue', methods=['POST'])
def watch_issue():
    # Verify webhook signature
    signature_check = verify_webhook_signature()
    if signature_check:
        return signature_check
    
    data = request.get_json()
    action = data.get('action')
    issue_data = data.get('data')
    issue_type = data.get('type')
    webhook_timestamp = data.get('webhookTimestamp')
    
    # Verify webhook timestamp
    import time
    current_timestamp = int(time.time() * 1000)
    if abs(current_timestamp - webhook_timestamp) > 60000:
        return jsonify({'error': 'Webhook timestamp is too old'}), 400
    
    if issue_type == 'Issue':
        linear_service.processIssueWebhook(action, issue_data)
    
    return jsonify({'message': 'Webhook processed successfully'}), 200

@app.route('/api/linear/projects', methods=['GET'])
def get_projects():
    try:
        projects = linear_service.fetchProjects()
        return jsonify(projects)
    except Exception as error:
        print('Error fetching projects:', error)
        return jsonify({'error': 'Failed to fetch projects'}), 500

@app.route('/api/linear/project/<project_id>/statuses', methods=['GET'])
def get_project_statuses(project_id):
    try:
        statuses = linear_service.fetchProjectStatuses(project_id)
        return jsonify(statuses)
    except Exception as error:
        print('Error fetching project statuses:', error)
        return jsonify({'error': 'Failed to fetch project statuses'}), 500

@app.route('/api/linear/issues', methods=['GET'])
def get_issues():
    try:
        project_id = request.args.get('projectId')
        issues = linear_service.fetchIssues(project_id)
        return jsonify(issues)
    except Exception as error:
        print('Error fetching issues:', error)
        return jsonify({'error': 'Failed to fetch issues'}), 500

@app.route('/api/linear/issues/<issue_id>', methods=['PATCH'])
def update_issue(issue_id):
    try:
        update_data = request.get_json()
        updated_issue = linear_service.updateIssue(issue_id, update_data)
        
        if updated_issue:
            return jsonify(updated_issue)
        else:
            return jsonify({'error': 'Issue not found or update failed'}), 404
    except Exception as error:
        print('Error updating issue:', error)
        return jsonify({'error': 'Failed to update issue'}), 500

@app.errorhandler(Exception)
def handle_error(error):
    print(f"Error: {str(error)}")
    return jsonify({'error': 'Something went wrong!'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    print(f'Server running on port {port}')
    app.run(host='0.0.0.0', port=port, debug=True)
