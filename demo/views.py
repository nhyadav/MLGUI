from asyncio.log import logger
from django.shortcuts import render
import logging
# Create your views here.

logger = logging.getLogger(__name__)


def home(request):
    logger.info("This is First Logger setup-in Django.")
    logger.debug("this is debugging")
    return render(request, 'index.html')