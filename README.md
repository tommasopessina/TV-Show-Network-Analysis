# TV-Show Social Network Analysis

import necessary for calling external function


```python
import networkx as nx
import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
```

Create the graph, read the file and add edge and nodes 


```python
graph = nx.Graph()
graph_edge_list=nx.Graph()

with open('fb-pages-tvshowEDGE.csv') as f:
    f.readline()
    for line in f:
        source, destination = line.strip().split(',') #tuple
        graph.add_edge(source,destination)
        
print('Number of nodes: {} - Number of links:{}'.format(graph.order(),graph.size()))
```

    Number of nodes: 3892 - Number of links:17261
    

Read the nodes labels from file and relabel the graph nodes


```python
labels = {}

df = pd.read_csv("fb-pages-tvshowNODES.csv")

df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

for index, row in df.iterrows():
    tmp = int(row["Id"])
    labels[str(tmp)] = str(row["name"])
    
graph = nx.relabel_nodes(graph, labels)

pos = nx.spring_layout(graph)
```

Run some statistics over the network


```python
graph_degree=list(dict(graph.degree()).values())

print("Mean degree: "+ str(np.mean(graph_degree)))
print("Median degree: " + str(np.median(graph_degree))) 
print("Standard Deviation: "+ str(np.std(graph_degree)))
print("Max degree value: " + str(np.max(graph_degree)))
print("Min degree value: " + str(np.min(graph_degree)))
```

    Mean degree: 8.932435930623868
    Median degree: 5.0
    Standard Deviation: 12.658862276836015
    Max degree value: 126
    Min degree value: 1
    

A first simple degree analysis


```python
def getKey(label):
    keys = list(labels.keys())
    vals = list(labels.values())
    key = keys[vals.index(label)]
    return key

print("Modern Family has degree " + str(graph.degree["Modern Family"]))
print("The Office has degree " + str(graph.degree["The Office"]))
print("Hell's Kitchen has degree " + str(graph.degree["Hell's Kitchen"]))
print("The Big Bang Theory has degree " + str(graph.degree["The Big Bang Theory"]))
```

    Modern Family has degree 86
    The Office has degree 45
    Hell's Kitchen has degree 94
    The Big Bang Theory has degree 36
    

# Degree analysis
Define some constant


```python
max_element = 15
```

## Degree Centrality
This is based on the assumption that important nodes have many connections


```python
centrality = nx.degree_centrality(graph)

sort_orders = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

degree_centr = []

count=0
for i in sort_orders:
    if count==max_element:
        break
    count=count+1
    print(i[0], i[1])
    degree_centr.append(i[0])
```

    Home & Family 0.03262558259968928
    Queen of the South 0.03262558259968928
    So You Think You Can Dance 0.027964785085447953
    MasterChef 0.027705851890212324
    Glee 0.026929052304505437
    New Girl 0.02641118591403418
    Family Guy 0.02615225271879855
    The Simpsons 0.02589331952356292
    Dancing with the Stars 0.02589331952356292
    Bones 0.025116519937856033
    Brooklyn Nine-Nine 0.02459865354738477
    Bob's Burgers 0.02459865354738477
    Hell's Kitchen 0.024339720352149142
    Sleepy Hollow 0.023045054375970996
    Modern Family 0.02226825479026411
    

## Closeness Centrality
This is based on the assumption that important nodes are close to other nodes.


```python
close_centrality = nx.closeness_centrality(graph) 

sort_orders = sorted(close_centrality.items(), key=lambda x: x[1], reverse=True)

close_centr = []

count=0
for i in sort_orders:
    if count==max_element:
        break
    count=count+1
    print(i[0], i[1])
    close_centr.append(i[0])
```

    Queen of the South 0.26575832645196806
    Home & Family 0.2640864332603939
    Access 0.2594383984952304
    The Tonight Show Starring Jimmy Fallon 0.2537617451869374
    The Voice 0.2489043567929879
    The List 0.2461440407903123
    America's Got Talent 0.2456274247917064
    The Insider 0.2449575034885196
    The Biggest Loser 0.24326026706979087
    Entourage 0.242268364594442
    Parenthood 0.24125437281359322
    Jay Leno 0.24074304949507544
    Parks and Recreation 0.23992048207740574
    New Girl 0.23951872984371123
    Extra 0.23779323933255342
    

## Betweenness Centrality
It assumes that important nodes connect other nodes


```python
bet_centrality = nx.betweenness_centrality(graph, normalized = True,  
                                              endpoints = False)

sort_orders = sorted(close_centrality.items(), key=lambda x: x[1], reverse=True)

between_centr = []

count=0
for i in sort_orders:
    if count==max_element:
        break
    count=count+1
    print(i[0], i[1])
    between_centr.append(i[0])
```

    Queen of the South 0.26575832645196806
    Home & Family 0.2640864332603939
    Access 0.2594383984952304
    The Tonight Show Starring Jimmy Fallon 0.2537617451869374
    The Voice 0.2489043567929879
    The List 0.2461440407903123
    America's Got Talent 0.2456274247917064
    The Insider 0.2449575034885196
    The Biggest Loser 0.24326026706979087
    Entourage 0.242268364594442
    Parenthood 0.24125437281359322
    Jay Leno 0.24074304949507544
    Parks and Recreation 0.23992048207740574
    New Girl 0.23951872984371123
    Extra 0.23779323933255342
    

## Page Rank
Page Rank Algorithm was developed by Google founders to measure the importance of webpages from the hyperlink network structure. Page Rank assigns a score of importance to each node. Important nodes are those with many inlinks from important pages.


```python
pr = nx.pagerank(graph, alpha = 0.8)

sort_orders = sorted(close_centrality.items(), key=lambda x: x[1], reverse=True)

page_rank = []

count=0
for i in sort_orders:
    if count==max_element:
        break
    count=count+1
    print(i[0], i[1])
    page_rank.append(i[0])
```

    Queen of the South 0.26575832645196806
    Home & Family 0.2640864332603939
    Access 0.2594383984952304
    The Tonight Show Starring Jimmy Fallon 0.2537617451869374
    The Voice 0.2489043567929879
    The List 0.2461440407903123
    America's Got Talent 0.2456274247917064
    The Insider 0.2449575034885196
    The Biggest Loser 0.24326026706979087
    Entourage 0.242268364594442
    Parenthood 0.24125437281359322
    Jay Leno 0.24074304949507544
    Parks and Recreation 0.23992048207740574
    New Girl 0.23951872984371123
    Extra 0.23779323933255342
    

### Here we can see the different result jointly


```python
df = pd.DataFrame()
df["Degree Centrality"] = degree_centr
df["Closeness Centrality"] = close_centr
df["Betweenness Centrality"] = between_centr
df["Page Rank"] = page_rank
df.style
```




<style  type="text/css" >
</style><table id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Degree Centrality</th>        <th class="col_heading level0 col1" >Closeness Centrality</th>        <th class="col_heading level0 col2" >Betweenness Centrality</th>        <th class="col_heading level0 col3" >Page Rank</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row0_col0" class="data row0 col0" >Home & Family</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row0_col1" class="data row0 col1" >Queen of the South</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row0_col2" class="data row0 col2" >Queen of the South</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row0_col3" class="data row0 col3" >Queen of the South</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row1_col0" class="data row1 col0" >Queen of the South</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row1_col1" class="data row1 col1" >Home & Family</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row1_col2" class="data row1 col2" >Home & Family</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row1_col3" class="data row1 col3" >Home & Family</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row2_col0" class="data row2 col0" >So You Think You Can Dance</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row2_col1" class="data row2 col1" >Access</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row2_col2" class="data row2 col2" >Access</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row2_col3" class="data row2 col3" >Access</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row3_col0" class="data row3 col0" >MasterChef</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row3_col1" class="data row3 col1" >The Tonight Show Starring Jimmy Fallon</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row3_col2" class="data row3 col2" >The Tonight Show Starring Jimmy Fallon</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row3_col3" class="data row3 col3" >The Tonight Show Starring Jimmy Fallon</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row4_col0" class="data row4 col0" >Glee</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row4_col1" class="data row4 col1" >The Voice</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row4_col2" class="data row4 col2" >The Voice</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row4_col3" class="data row4 col3" >The Voice</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row5_col0" class="data row5 col0" >New Girl</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row5_col1" class="data row5 col1" >The List</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row5_col2" class="data row5 col2" >The List</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row5_col3" class="data row5 col3" >The List</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row6_col0" class="data row6 col0" >Family Guy</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row6_col1" class="data row6 col1" >America's Got Talent</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row6_col2" class="data row6 col2" >America's Got Talent</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row6_col3" class="data row6 col3" >America's Got Talent</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row7_col0" class="data row7 col0" >The Simpsons</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row7_col1" class="data row7 col1" >The Insider</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row7_col2" class="data row7 col2" >The Insider</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row7_col3" class="data row7 col3" >The Insider</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row8_col0" class="data row8 col0" >Dancing with the Stars</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row8_col1" class="data row8 col1" >The Biggest Loser</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row8_col2" class="data row8 col2" >The Biggest Loser</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row8_col3" class="data row8 col3" >The Biggest Loser</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row9_col0" class="data row9 col0" >Bones</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row9_col1" class="data row9 col1" >Entourage</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row9_col2" class="data row9 col2" >Entourage</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row9_col3" class="data row9 col3" >Entourage</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row10_col0" class="data row10 col0" >Brooklyn Nine-Nine</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row10_col1" class="data row10 col1" >Parenthood</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row10_col2" class="data row10 col2" >Parenthood</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row10_col3" class="data row10 col3" >Parenthood</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row11_col0" class="data row11 col0" >Bob's Burgers</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row11_col1" class="data row11 col1" >Jay Leno</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row11_col2" class="data row11 col2" >Jay Leno</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row11_col3" class="data row11 col3" >Jay Leno</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row12_col0" class="data row12 col0" >Hell's Kitchen</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row12_col1" class="data row12 col1" >Parks and Recreation</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row12_col2" class="data row12 col2" >Parks and Recreation</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row12_col3" class="data row12 col3" >Parks and Recreation</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row13_col0" class="data row13 col0" >Sleepy Hollow</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row13_col1" class="data row13 col1" >New Girl</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row13_col2" class="data row13 col2" >New Girl</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row13_col3" class="data row13 col3" >New Girl</td>
            </tr>
            <tr>
                        <th id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row14_col0" class="data row14 col0" >Modern Family</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row14_col1" class="data row14 col1" >Extra</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row14_col2" class="data row14 col2" >Extra</td>
                        <td id="T_e12d24b6_27f4_11eb_9943_54e1ad281be9row14_col3" class="data row14 col3" >Extra</td>
            </tr>
    </tbody></table>



# Isolation and Connectivity


```python
print(list(nx.isolates(graph)))
```

    []
    


```python
print(nx.is_connected(graph))
print(nx.number_connected_components(graph))
```

    True
    1
    

So, we have no isolated component which mean that our graph is connected and it has only one (big) connected component 

# ECDF in linear e logscale
Here we analyse the degree distribution of our network and turns out that our graph is a scale-free network (like most of the real-world case) because it follow a power law degree distribution (heavy-tail distribution).


```python
# ECDF linear scale
cdf = ECDF(graph_degree)
x = np.unique(graph_degree)
y = cdf(x)
fig_cdf = plt.figure(figsize=(8,4))
axes = fig_cdf.gca()
axes.plot(x,y,marker='o',ms=6, linestyle='None')
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('ECDF TV Show', size = 20)
```




    Text(0, 0.5, 'ECDF TV Show')




![png](output_28_1.png)



```python
# ECDF loglog scale
cdf = ECDF(graph_degree)
x = np.unique(graph_degree)
y = cdf(x)
fig_cdf = plt.figure(figsize=(8,4))
axes = fig_cdf.gca()
axes.loglog(x,y,marker='o',ms=8, linestyle='--')
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('ECDF TV Show', size = 20)
```




    Text(0, 0.5, 'ECDF TV Show')




![png](output_29_1.png)



```python
# ECCDF
cdf = ECDF(graph_degree)
x = np.unique(graph_degree)
y = cdf(x)
fig_cdf = plt.figure(figsize=(8,4))
axes = fig_cdf.gca()
axes.loglog(x,1-y,marker='o',ms=8, linestyle='--')
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('ECDF TV Show', size = 20)
```




    Text(0, 0.5, 'ECDF TV Show')




![png](output_30_1.png)


Now we can compare it with an Erdos-Renyi random network model to realize if the phenomenon carries information of if it is random.
First we calculate the densitiy for our starting network and we see that, as expected, is low.


```python
density = nx.density(graph)
print('Density: {}'.format(density))
```

    Density: 0.0023129041767539792
    

We use the density as parameter p of the Erdos-Renyi model, i.e. the probability that each pair of N labeled nodes is connected. The number of nodes of the two network must be equal and the number of links should be similiar.


```python
p = density

random_graph = nx.fast_gnp_random_graph(graph.order(),p)

print('Number of nodes: {} - Number of links:{} -> TV-Show netowrk'.format(graph.order(),graph.size()))
print('Number of nodes: {} -> random network'.format(random_graph.order()))
print('Number of links: {} -> random network'.format(random_graph.size()))

random_degree = list(dict(random_graph.degree()).values())
```

    Number of nodes: 3863 - Number of links:17253 -> TV-Show netowrk
    Number of nodes: 3863 -> random network
    Number of links: 17156 -> random network
    


```python
cdf = ECDF(graph_degree)
x = np.unique(graph_degree)
y = cdf(x)

cdf_random = ECDF(random_degree)
x_random = np.unique(random_degree)
y_random = cdf_random(x_random)

fig_cdf_fb = plt.figure(figsize=(8,4))
axes = fig_cdf_fb.gca()
axes.set_xscale('log')
axes.set_yscale('log')
axes.loglog(x,1-y,marker='o',ms=8, linestyle='--')
axes.loglog(x_random,1-y_random,marker='+',ms=10, linestyle='--')
axes.set_xlabel('Degree',size=20)
axes.set_ylabel('ECCDF', size = 20)
```




    Text(0, 0.5, 'ECCDF')




![png](output_35_1.png)


We can see that they behave in a similar manner

# Hubs
Since we saw that we are talking about a scale-free network we can also say if there exist some hubs, by deeper analyse the degree distribution.


```python
percentile_99 = np.percentile(graph_degree,99)
print("99 percentile: ", percentile_99)

hub_nodi = [k for k,v in dict(graph.degree()).items() if v>= percentile_99]

print("Number of hubs: ",len(hub_nodi))
#print(list(hub_nodi))

dfh = pd.DataFrame()
dfh["Hub/Tv-show/node name"] = hub_nodi
dfh.style
```

    99 percentile:  65.0
    Number of hubs:  40
    




<style  type="text/css" >
</style><table id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Hub/Tv-show/node name</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row0_col0" class="data row0 col0" >American Grit</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row1_col0" class="data row1 col0" >Wayward Pines</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row2_col0" class="data row2 col0" >MasterChef</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row3_col0" class="data row3 col0" >The Grinder</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row4_col0" class="data row4 col0" >FOX Teen Choice Awards</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row5_col0" class="data row5 col0" >Rosewood</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row6_col0" class="data row6 col0" >Scream Queens</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row7_col0" class="data row7 col0" >New Girl</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row8_col0" class="data row8 col0" >Gotham</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row9_col0" class="data row9 col0" >Family Guy</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row10_col0" class="data row10 col0" >Grandfathered</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row11_col0" class="data row11 col0" >Empire</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row12_col0" class="data row12 col0" >Second Chance</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row13_col0" class="data row13 col0" >So You Think You Can Dance</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row14_col0" class="data row14 col0" >The Simpsons</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row15_col0" class="data row15 col0" >Glee</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row16_col0" class="data row16 col0" >MasterChef Junior</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row17_col0" class="data row17 col0" >Bones</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row18_col0" class="data row18 col0" >Sleepy Hollow</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row19_col0" class="data row19 col0" >Lucifer</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row20_col0" class="data row20 col0" >Hell's Kitchen</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row21_col0" class="data row21 col0" >24: Legacy</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row22_col0" class="data row22 col0" >My Kitchen Rules</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row23_col0" class="data row23 col0" >Brooklyn Nine-Nine</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row24_col0" class="data row24 col0" >Hotel Hell</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row25_col0" class="data row25 col0" >The Last Man on Earth</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row26_col0" class="data row26 col0" >Bob's Burgers</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row27_col0" class="data row27 col0" >The Bachelorette</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row28_col0" class="data row28 col0" >Castle</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row29_col0" class="data row29 col0" >Dancing with the Stars</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row30_col0" class="data row30 col0" >The Middle</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row31_col0" class="data row31 col0" >Modern Family</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row32_col0" class="data row32 col0" >The View</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row33_col0" class="data row33 col0" >Once Upon a Time</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row34_col0" class="data row34 col0" >Revenge</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row35_col0" class="data row35 col0" >Suburgatory</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row36_col0" class="data row36 col0" >Rookie Blue</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row37_col0" class="data row37 col0" >Home & Family</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row38_col0" class="data row38 col0" >Queen of the South</td>
            </tr>
            <tr>
                        <th id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9level0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_e2a3a40d_27f4_11eb_bfda_54e1ad281be9row39_col0" class="data row39 col0" >tagesschau</td>
            </tr>
    </tbody></table>



# Conclusion

From our degree analysis we see that different method (basing on different assumption) gives us different result.
From last analysis, the one about hubs, we can see that there are some - let's say - important node in our network and not all of them appear in our degree analysis (and vice versa).


```python

```
