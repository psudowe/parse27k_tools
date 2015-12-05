"""Git helper functions
"""

import subprocess, os


def current_revision(path, short=False):
    """takes a 'path' and returns a string representing the
    current HEAD git revision.
    Optionally, only print a short revision hash.
    """
    gitdir = os.path.join(path, '.git')

    if not os.path.isdir(gitdir):
        return None

    if short:
        cmd = ['git', '--git-dir', gitdir, '--work-tree', path,
               'rev-parse', '--short', 'HEAD'
              ]
    else:
        cmd = ['git', '--git-dir', gitdir, '--work-tree', path,
               'rev-parse', 'HEAD'
              ]

    rev = subprocess.check_output(cmd).strip()

    # little workaround to avoid output to stdin/stderr
    # the returncode is either 0 - or the exception is raised
    try:
        subprocess.check_output(['git', '--git-dir', gitdir,
                                 '--work-tree', path,
                                 'diff', '--exit-code'
                                ])
        dirty = False
    except subprocess.CalledProcessError as e:
        dirty = e.returncode

    if dirty:
        print 'WARNING: DIRTY GIT REPOSITORY...'
        rev = 'DIRTY_GIT_REPO__' + rev

    return rev
