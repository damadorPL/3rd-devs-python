import asyncio
from typing import Optional, Dict, Any, List, Set
from linear_api import LinearClient
from openai_service import OpenAIService

class ProjectAssignment:
    def __init__(self, thoughts: str, name: str, id: str):
        self._thoughts = thoughts
        self.name = name
        self.id = id

class LinearService:
    def __init__(self, api_key: str):
        self.client = LinearClient(api_key)
        self.openAIService = OpenAIService()
        self.validProjectIds: Set[str] = set()
        asyncio.create_task(self.initializeValidProjectIds())
    
    async def initializeValidProjectIds(self):
        projects = await self.fetchProjects()
        for project in projects.get('nodes', []):
            self.validProjectIds.add(project['id'])
    
    def processIssueWebhook(self, action: str, data: Dict[str, Any]) -> None:
        issue_details = self.extractIssueDetails(data)
        
        if action == 'create':
            print(f"New issue created:", issue_details)
            if not issue_details.get('project'):
                asyncio.create_task(
                    self.assignProjectToIssue(
                        issue_details['id'],
                        issue_details['title'],
                        issue_details['description']
                    )
                )
        elif action == 'update':
            print(f"Issue updated:", issue_details)
        elif action == 'remove':
            print(f"Issue removed:", {
                'id': issue_details['id'],
                'title': issue_details['title']
            })
        else:
            print(f"Unhandled action: {action}", issue_details)
    
    async def assignProjectToIssue(self, issue_id: str, title: str, description: str):
        try:
            assignment = await self.openAIService.assignProjectToTask(title, description)
            project_id = assignment.id if assignment.id in self.validProjectIds else "ad799a5f-259c-4ff1-9387-efb949a56508"
            
            await self.updateIssue(issue_id, {'projectId': project_id})
            print(f"Assigned issue {issue_id} to project {assignment.name} ({project_id})")
        except Exception as error:
            print(f"Error assigning project to issue {issue_id}:", error)
    
    def extractIssueDetails(self, data: Dict[str, Any]) -> Dict[str, Any]:
        description = data.get('description', '')
        truncated_description = description[:100] + ('...' if len(description) > 100 else '')
        
        return {
            'id': data.get('id'),
            'title': data.get('title'),
            'description': truncated_description,
            'priority': data.get('priority'),
            'status': data.get('status', {}).get('name') if data.get('status') else None,
            'assignee': data.get('assignee', {}).get('name') if data.get('assignee') else None,
            'team': data.get('team', {}).get('name') if data.get('team') else None,
            'project': data.get('project', {}).get('name') if data.get('project') else None,
            'createdAt': data.get('createdAt'),
            'updatedAt': data.get('updatedAt')
        }
    
    async def fetchProjects(self) -> Dict[str, Any]:
        return await self.client.projects()
    
    async def fetchProjectDetails(self, project_id: str) -> Optional[Dict[str, Any]]:
        return await self.client.project(project_id)
    
    async def fetchProjectStatuses(self, project_id: str) -> List[str]:
        project = await self.fetchProjectDetails(project_id)
        if not project:
            return []
        
        teams = await project.teams()
        if not teams.get('nodes'):
            return []
        
        team = teams['nodes'][0]
        states = await team.states()
        return [state['name'] for state in states.get('nodes', [])]
    
    async def fetchIssues(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        filter_params = {}
        if project_id:
            filter_params['project'] = {'id': {'eq': project_id}}
        
        return await self.client.issues(filter=filter_params)
    
    async def updateIssue(self, issue_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            result = await self.client.updateIssue(issue_id, update_data)
            if result.get('success') and result.get('issue'):
                print(f"Issue updated successfully: {issue_id}")
                return result['issue']
            else:
                print(f"Failed to update issue: {issue_id}")
                return None
        except Exception as error:
            print(f"Error updating issue {issue_id}:", error)
            raise error
