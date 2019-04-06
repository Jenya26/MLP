import os
import traceback
from importlib import import_module

from assertion import AssertionFail

TESTS_DIR_NAME = "tests"

CURRENT_DIRECTORY = os.path.abspath(os.curdir)


def run_test(test_name, test, report):
    result = {
        'value': [[test_name, '-']],
        'offset': 2
    }
    report += [result]
    try:
        test()
        result['value'][0] += ['ok']
        return True, 1, 1
    except:
        result['value'] += [{
            'value': traceback.format_exc(),
            'offset': 2
        }]
    result['value'][0] += ['fail']
    return False, 0, 1


def run_test_suit(test_path, report):
    test_suit_path_without_extension = os.path.splitext(test_path)[0]
    test_suit_name = os.path.basename(test_suit_path_without_extension)
    module_path = os.path.relpath(test_suit_path_without_extension, CURRENT_DIRECTORY).replace('/', '.')
    test_suit = import_module(module_path)
    status = True
    passed_tests = 0
    all_tests = 0
    result = {
        'value': [[test_suit_name, '-']],
        'offset': 2
    }
    if hasattr(test_suit, '__all__'):
        for test_name in test_suit.__all__:
            if not hasattr(test_suit, test_name):
                continue
            test = getattr(test_suit, test_name)
            if callable(test):
                test_status, test_passed_tests, test_all_tests = run_test(test_name, test, result['value'])
                status = status and test_status
                passed_tests += test_passed_tests
                all_tests += test_all_tests
    result['value'][0] += ['ok' if status else 'fail']
    result['value'][0] += [f"{passed_tests}/{all_tests}"]
    result['value'] += '\n'
    report += [result]
    return status, passed_tests, all_tests


def run_tests_suit(tests_path, report):
    group_name = os.path.basename(os.path.dirname(tests_path))
    status = True
    passed_tests = 0
    all_tests = 0
    result = {
        'value': [[group_name, '-']],
        'offset': 2
    }
    for test_suit_path in os.listdir(tests_path):
        test_suit_path = os.path.abspath(os.path.join(tests_path, test_suit_path))
        if os.path.isfile(test_suit_path):
            test_suit_status, test_suit_passed_tests, test_suit_all_tests = run_test_suit(test_suit_path, result['value'])
            status = status and test_suit_status
            passed_tests += test_suit_passed_tests
            all_tests += test_suit_all_tests
    result['value'][0] += ['ok' if status else 'fail']
    result['value'][0] += [f"{passed_tests}/{all_tests}"]
    report += [result]
    return status, passed_tests, all_tests


def find_tests_suit(project_dir, report):
    project_name = os.path.basename(os.path.splitext(project_dir)[0])
    status = True
    passed_tests = 0
    all_tests = 0
    result = {
        'value': [[project_name, '-']],
        'offset': 0
    }
    for dir_path in os.listdir(project_dir):
        tests_path = os.path.abspath(os.path.join(dir_path, TESTS_DIR_NAME))
        if os.path.exists(tests_path):
            tests_suit_status, tests_suit_passed_tests, tests_suit_all_tests = run_tests_suit(tests_path,
                                                                                              result['value'])
            status = status and tests_suit_status
            passed_tests += tests_suit_passed_tests
            all_tests += tests_suit_all_tests
    result['value'][0] += ['ok' if status else 'fail']
    result['value'][0] += [f"{passed_tests}/{all_tests}"]
    report += [result]
    report += [{
        'value': f'Tests passed {passed_tests} of {all_tests}'
    }]


default_state = {
    'prevType': None,
    'offset': 0,
    'newLine': False,
    'lastOffset': 0
}


def next_line(state):
    state['newLine'] = True
    state['lastOffset'] = state['offset']
    return '\n' + ' ' * state['offset']


def next_char(ch, state):
    state['newLine'] = False
    return ch


def fix_offset(state):
    offset = 0
    if state['newLine']:
        offset = state['offset'] - state['lastOffset']
    state['lastOffset'] = state['offset']
    return ' ' * offset


def get_report(report, state=None):
    if state is None:
        state = default_state
    result = ""
    if isinstance(report, list):
        result += get_report(report[0], state)
        for report_part in report[1:]:
            if state['prevType'] == 'list':
                result += next_line(state)
            result += get_report(report_part, state)
        state['prevType'] = 'list'
        return result
    if isinstance(report, dict):
        if state['prevType'] == 'dict':
            result += next_line(state)
        dict_offset = 0 if 'offset' not in report else report['offset']
        state['offset'] += dict_offset
        result += get_report(report['value'], state)
        state['offset'] -= dict_offset
        state['prevType'] = 'dict'
        return result
    # if state['prevType'] is None:
    #     result += ' ' * state['offset']
    result += fix_offset(state)
    if state['prevType'] == 'text':
        result += ' '
    for ch in report:
        if ch == '\n':
            result += next_line(state)
            continue
        result += next_char(ch, state)
    state['prevType'] = 'text'
    return result


if __name__ == "__main__":
    report = []
    find_tests_suit(CURRENT_DIRECTORY, report)
    print(get_report(report))
