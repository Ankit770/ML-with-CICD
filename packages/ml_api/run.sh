export IS_DEBUG=${DEBUG:-false}
exec gunicorn -b :{PORT:-8000} --access-logfile - --error-logfile - run:application