from datetime import datetime
import re
from typing import List, Dict, Any, Optional
from tabulate import tabulate
from colorama import Fore, Style, init

# Initialize colorama
init()

def current_datetime() -> str:
    """Get current date and time in YYYY-MM-DD HH:mm format."""
    now = datetime.now()
    return now.strftime('%Y-%m-%d %H:%M')

def format_datetime(date: datetime, include_time: bool = True) -> str:
    """Format datetime object to string.

    Args:
        date: datetime object to format
        include_time: whether to include time in the output

    Returns:
        Formatted date string in YYYY-MM-DD HH:mm or YYYY-MM-DD format
    """
    if include_time:
        return date.strftime('%Y-%m-%d %H:%M')
    return date.strftime('%Y-%m-%d')

def get_result(content: str, tag_name: str) -> Optional[str]:
    """Extract content between XML-like tags.

    Args:
        content: string containing XML-like tags
        tag_name: name of the tag to extract content from

    Returns:
        Content between the specified tags, or None if not found
    """
    pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None

def display_results_as_table(results: List[Dict[str, Any]]) -> None:
    """Display test results in a formatted table.

    Args:
        results: List of test result dictionaries
    """
    print(results)

    # Prepare table data
    headers = [
        f"{Style.BRIGHT}Query{Style.RESET_ALL}",
        f"{Style.BRIGHT}Variables{Style.RESET_ALL}",
        f"{Style.BRIGHT}Result{Style.RESET_ALL}"
    ]

    table_data = []
    for result in results:
        # Format result output
        if result['success']:
            result_output = f"{Fore.GREEN}[PASS] {result['response']['output']}{Style.RESET_ALL}"
        else:
            error_msg = result['error'].split('Stack Trace')[0].strip()
            result_output = f"{Fore.RED}[ERROR] {error_msg}\n\n-- {result['response']['output']}{Style.RESET_ALL}"

        # Prepare row data
        row = [
            result['testCase']['vars'].get('query', ''),
            str({k: v for k, v in result['testCase']['vars'].items() if k != 'query'}),
            result_output
        ]
        table_data.append(row)

    # Print table
    print(tabulate(
        table_data,
        headers=headers,
        tablefmt='grid',
        colalign=('left', 'left', 'left'),
        maxcolwidths=[20, 65, 95]
    ))

    # Add summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests

    print(f"\n{Style.BRIGHT}Summary:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Passed: {passed_tests}{Style.RESET_ALL}")
    print(f"{Fore.RED}Failed: {failed_tests}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Total: {total_tests}{Style.RESET_ALL}")

# Example usage:
if __name__ == "__main__":
    # Test current_datetime
    print("Current datetime:", current_datetime())

    # Test format_datetime
    now = datetime.now()
    print("Formatted datetime with time:", format_datetime(now))
    print("Formatted datetime without time:", format_datetime(now, include_time=False))

    # Test get_result
    test_content = "<test>This is a test</test>"
    print("Extracted result:", get_result(test_content, "test"))

    # Test display_results_as_table with sample data
    sample_results = [
        {
            'success': True,
            'response': {'output': 'Test passed'},
            'testCase': {'vars': {'query': 'Test query', 'param': 'value'}},
            'error': ''
        },
        {
            'success': False,
            'response': {'output': 'Test failed'},
            'testCase': {'vars': {'query': 'Failed query', 'param': 'value'}},
            'error': 'Error message\nStack Trace: ...'
        }
    ]
    display_results_as_table(sample_results)
