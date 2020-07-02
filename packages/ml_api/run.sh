export IS_DEBUG=${DEBUG:-false}
exec gunicorn -b :${PORT:-8888} --access-logfile - --error-logfile - run:application