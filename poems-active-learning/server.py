import asyncio
import sys
import traceback

from training_manager import TrainingManager
from aiohttp import web
import json
import pandas


async def start_train_wrapper(app):
    def callback(f):
        try:
            f.result()
        except:
            traceback.print_exc()
            print("Shutdown")
            asyncio.get_running_loop().create_task(app.shutdown())

    asyncio.wrap_future(app['manager'].start_training()).add_done_callback(callback)


async def status(request):
    return web.json_response(request.app['manager'].get_status_object())


async def label_batch(request):
    if request.app['manager'].is_training():
        return web.json_response({'error': 'Model is training'}, status=500)
    elif request.app['manager'].classification_output is None:
        return web.json_response({'error': 'Model was not evaluated'}, status=500)
    else:
        frame = request.app['manager'].get_label_batch()
        return web.json_response({
            'date': request.app['manager'].status_object['enddate'],
            'docs': frame.to_dict('records')
        })


async def start(request):
    if request.app['manager'].is_training():
        return web.json_response({'error': 'Model is training'}, status=500)
    else:
        await start_train_wrapper(request.app)
        return web.json_response({'message': 'Model started training'})


async def upload_batch(request):
    try:
        body = await request.json()
        if 'labeled_docs' not in body:
            return web.Response(status=400)

        labeled_docs = body['labeled_docs']
        df = pandas.DataFrame.from_records(labeled_docs, columns=['filename', 'labels']).dropna()
        app['manager'].add_labeled(df)
        return web.json_response({'message': 'Labeled documents added'})
    except json.JSONDecodeError:
        return web.Response(status=400)


async def index(request):
    raise web.HTTPFound('/static/labelview.html')


async def shutdown_handler(app):
    sys.exit(1)

app = web.Application()
app['manager'] = TrainingManager(document_dir='./poems', labeled_documents_file='./labeled_documents.tsv')

app.router.add_get('/api/status', status)
app.router.add_get('/api/starttrain', start)
app.router.add_get('/api/labelbatch', label_batch)
app.router.add_post('/api/uploadbatch', upload_batch)
app.router.add_static('/static', 'static')
app.router.add_get('/', index)
app.on_startup.append(start_train_wrapper)
app.on_shutdown.append(shutdown_handler)

web.run_app(app, host='127.0.0.1', port=9999 if len(sys.argv) < 2 else int(sys.argv[1]))
