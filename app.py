import base64
from io import BytesIO
import time

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
import requests
from get_index import get_index

# Dash component wrappers
def Row(children=None, **kwargs):
    return html.Div(children, className="row", **kwargs)


def Column(children=None, width=1, **kwargs):
    nb_map = {
        1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six',
        7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve'}

    return html.Div(children, className=f"{nb_map[width]} columns", **kwargs)


# plotly.py helper functions
def pil_to_b64(im, enc="png"):
    io_buf = BytesIO()
    im.save(io_buf, format=enc)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def pil_to_fig(im, showlegend=False, title=None):
    img_width, img_height = im.size
    fig = go.Figure()
    # This trace is added to help the autoresize logic work.
    fig.add_trace(go.Scatter(
        x=[img_width * 0.05, img_width * 0.95],
        y=[img_height * 0.95, img_height * 0.05],
        showlegend=False, mode="markers", marker_opacity=0,
        hoverinfo="none", legendgroup='Image'))

    fig.add_layout_image(dict(
        source=pil_to_b64(im), sizing="stretch", opacity=1, layer="below",
        x=0, y=0, xref="x", yref="y", sizex=img_width, sizey=img_height))

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width])

    fig.update_yaxes(
        showgrid=False, visible=False,
        scaleanchor="x", scaleratio=1,
        range=[img_height, 0])

    fig.update_layout(title=title, showlegend=showlegend,
                      # annotations=[
                          # go.layout.Annotation(          # see https://community.plotly.com/t/how-to-include-a-source-footer-on-a-table-image/20220/3
                          #     showarrow=False,
                          #     text='Index: ' + index + ' m³',
                          #     font = {"size": 16},
                          #     bgcolor = 'rgb(255, 255, 255)',
                          #     bordercolor = 'rgb(0, 0, 0)',
                          #     borderpad=10,
                          #     width=150,
                          #     xanchor='right',
                          #     x=1,
                          #     xshift=-10,
                          #     yanchor='bottom',
                          #     y=img_height / 2
                          # )]
                     )

    return fig


def add_bbox(fig, x1, y1, x2, y2,
             showlegend=True, name=None, color=None,
             opacity=0.5, group=None, text=None):
    fig.add_trace(go.Scatter(
        x=[x1, x2, x2, x1, x1],
        y=[y1, y1, y2, y2, y1],
        mode="lines",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        name=name,
        hoverlabel_namelength=0,
        text=text,
        legendgroup=group,
        showlegend=showlegend,
    ))

# get the name of the different classes
def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

# colors for visualization
colors_dict = ['#78635b', '#d49279', '#edca79', '#9fa85b', '#2f403a',
               '#298591', '#4d4694', '#4a1b5c', '#b52d95', '#4f2430',
                '#2f633f', '#3e595c']

# urls of the images of the test set loaded in a google drive
match_links_df = pd.read_csv('match_links.csv')
unmatch_links_df = pd.read_csv('unmatch_links.csv')
links_df = pd.concat((match_links_df, unmatch_links_df.iloc[:60, :]))
links_df = links_df.sample(frac=1)
RANDOM_URLS = links_df['link'].values
names_images = links_df['name'].values
zip_iterator = zip(links_df['link'], links_df['name'])
match_dict = dict(zip_iterator)

# load dataframe of predictions
test_pred_100_df = pd.read_csv('test_pred_20200918_165400_epoch100_box_counter.csv')
test_pred_100_df['counter'] = test_pred_100_df['counter'].apply(lambda x: x.replace('.jpg', ''))

# Start Dash
app = dash.Dash(__name__)
server = app.server  # Expose the server variable for deployments

app.layout = html.Div(className='container', children=[
    Row(html.H1("YOLOv4 for Water Meter Reading")),
    Row(html.P("This app is to show the readings done on some water meters' images with a PyTorch YOLOv4 model.")),
    #Row(html.P("Input Image URL:")),
    Row(html.Br()),
    Row([
        Column(width=8, children=[
            dcc.Input(id='input-url', style={'width': '100%'},
                      disabled= True, placeholder='Insert URL...'
                      ),
        ], hidden=False),
        Column(html.Button("Predict", id='button-run', n_clicks=0), width=2),
        Column(html.Button("Random Image", id='button-random', n_clicks=0), width=2)
    ]),

    Row(dcc.Graph(id='model-output', style={"height": "70vh"})),

    Row([
        Column(width=7, children=[
            html.P('Non-maximum suppression (IoU):'),
            Row([
                Column(width=9, children=dcc.Slider(
                    id='slider-iou', min=0, max=1, step=0.05, value=0.4,
                    marks={0: '0', 0.1: '0.1', 0.2: '0.2', 0.3: '0.3', 0.4: '0.4',
                           0.5: '0.5', 0.6: '0.6', 0.7: '0.7', 0.8: '0.8',
                           0.9: '0.9', 1: '1'},
                    disabled=True)
                    ),
            ])
        ]),
        Column(width=5, children=[
            html.P('Confidence Threshold:'),
            dcc.Slider(
                id='slider-confidence', min=0, max=1, step=0.05, value=0.5,
                marks={0: '0', 0.1: '0.1', 0.2: '0.2', 0.3: '0.3', 0.4: '0.4',
                       0.5: '0.5', 0.6: '0.6', 0.7: '0.7', 0.8: '0.8',
                       0.9: '0.9', 1: '1'},
                disabled=True)
        ])
    ]),
    Row(html.Br()),
    Row(html.A(id="learn_more", children=html.Button("Learn More on the Dataset"),
               href="https://challengedata.ens.fr/challenges/30",
               target='_blank')
        )
])


@app.callback(
    [Output('button-run', 'n_clicks'),
     Output('input-url', 'value')],
    [Input('button-random', 'n_clicks')],
    [State('button-run', 'n_clicks')])
def randomize(random_n_clicks, run_n_clicks):
    return run_n_clicks+1, RANDOM_URLS[random_n_clicks%len(RANDOM_URLS)]


@app.callback(
    Output('model-output', 'figure'),
    [Input('button-run', 'n_clicks'),
     Input('input-url', 'n_submit'),
     Input('slider-iou', 'value'),
     Input('slider-confidence', 'value')],
    [State('input-url', 'value')])
def run_model(n_clicks, n_submit, iou, confidence, url):

    # compute the number of classes
    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    n_classes = file_len('_classes.txt')

    # namesfile
    namesfile = '_classes.txt'

    # load the image with the url
    try:
        img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    except:
        return go.Figure().update_layout(title='Incorrect URL')
    sized = img.resize((608, 608))

    # get the names of the _classes
    class_names = load_class_names(namesfile)

    # get name counter
    counter_name = match_dict[url]

    # get index
    temp_df = test_pred_100_df[test_pred_100_df['counter'] == counter_name]
    temp_df = temp_df.reset_index(drop = True)
    index = get_index(temp_df)

    if index == None:
        index = 'failure to predict'
    else:
        index = str(index) + ' m³'

    # display the image on the app
    fig = pil_to_fig(img, showlegend=True, title=f'Prediction of cubic meters: {index}')

    # plot the bounding boxes of the detected classes
    existing_classes = set()
    for i in range(temp_df.shape[0]):
        class_id = temp_df['cls_max_id'][i]
        confidence = temp_df['cls_max_conf'][i]
        x1 = temp_df['x1'][i]
        y1 = temp_df['y1'][i]
        x2 = temp_df['x2'][i]
        y2 = temp_df['y2'][i]

        # get the label of the class
        label = class_names[class_id]

        # only display legend when it's not in the existing classes
        showlegend = label not in existing_classes
        text = f"class={label}<br>confidence={confidence:.3f}"

        add_bbox(
            fig, x1, y1, x2, y2,
            opacity=1, group=label, name=label, color=colors_dict[class_id],
            showlegend=showlegend, text=text,
        )

        existing_classes.add(label  )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
