import pandas as pd

def get_cellInfo_CellNEST(
    ccc_pairs: pd.DataFrame,
    barcode_info: list(),
) -> defaultdict(dict), dict():
    """
    Return a dictionary as: [sender_cell][recvr_cell] = [(ligand gene, receptor gene, attention score), ...]
    for each pair of cells based on CellNEST detection. And a dictionary with cell_vs_index mapping.
    """
    """
    Parameters:
    ccc_pairs:  columns are ['from_cell', 'to_cell', 'ligand', 'receptor', 'edge_rank', 'component', 'from_id', 'to_id', 'attention_score']
    representing cell_barcode_sender, cell_barcode_receiver, ligand gene, receptor gene, 
    edge_rank, component_label, index_sender, index_receiver, attention_score
    barcode_info: list of [cell_barcode, coordinate_x, coordinates_y, -1]
    """
    cellPair_vs_genePair = defaultdict(dict)
    for i in range (0, len(ccc_pairs)):
        sender_cell_barcode = ccc_pairs['from_cell'] 
        rcvr_cell_barcode = ccc_pairs['to_cell']
        if sender_cell_barcode not in cellPair_vs_genePair or rcvr_cell_barcode not in cellPair_vs_genePair[sender_cell_barcode]:
            cellPair_vs_genePair[sender_cell_barcode][rcvr_cell_barcode] = []
        else:
            cellPair_vs_genePair[sender_cell_barcode][rcvr_cell_barcode].append([ccc_pairs['ligand'], ccc_pairs['receptor'], ccc_pairs['attention_score']])           

    cell_vs_index = dict()
    for i in range(0, len(barcode_info)):
        cell_vs_index[barcode_info[i][0]] = i

    return cellPair_vs_genePair, cell_vs_index

def get_cellEmb_geneEmb_pairs(
    cell_vs_index: dict(),
    barcode_info_gene: list(),
    X = np.array,
    X_g = np.array
) -> defaultdict(dict):
    """

    Parameters:
    cell_vs_index: dictionary with key = cell_barcode, value = index of that cell 
    barcode_info_gene: list of [cell's barcode, cell's X, cell's Y, -1, gene_node_index, gene_name]
    X = 2D np.array having row = cell index, column = feature dimension
    X_g = 2D np.array having row = gene node index, column = feature dimension
    """
    


