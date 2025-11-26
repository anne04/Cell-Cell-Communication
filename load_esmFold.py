import sys
sys.path.append('/cluster/home/t116508uhn/./.local/lib/python3.7/site-packages/')
sys.path.append('/cluster/tools/software/centos7/python/3.7.2/lib/python3.7/site-packages/')

from pathlib import Path
import torch
from esmFold_edited import ESMFold
import numpy as np
from Bio.PDB import PDBParser
import sys
from Bio.PDB import PDBParser, NeighborSearch, Selection
from collections import defaultdict


def calculate_bfactor_for_contact_residues_with_zeros(pdb_file, chain_a_id='A', chain_b_id='B', distance_threshold=5.0):
    # Initialize the PDB parser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    # Extract chains A and B
    chain_a = None
    chain_b = None
    for model in structure:
        chain_a = model[chain_a_id]
        chain_b = model[chain_b_id]
        break  # Assuming only one model is needed
    
    # Identify contact residues in chain B (within distance threshold to any residue in chain A)
    atoms_in_chain_a = Selection.unfold_entities(chain_a, 'A')  # List of all atoms in chain A
    atoms_in_chain_b = Selection.unfold_entities(chain_b, 'A')  # List of all atoms in chain B

    # Using NeighborSearch to find contacts
    ns = NeighborSearch(atoms_in_chain_a)
    contact_residues_in_chain_b = set()
    
    for atom in atoms_in_chain_b:
        neighbors = ns.search(atom.coord, distance_threshold, 'R')  # 'R' for residues
        if neighbors:
            contact_residues_in_chain_b.add(atom.get_parent().get_id())

    # To store B-factors for each residue in chain B (including zeros for non-contact residues)
    residue_bfactors = defaultdict(list)

    # Loop over all residues in chain B
    for residue in chain_b:
        residue_id = residue.get_id()
        if residue_id in contact_residues_in_chain_b:
            # Get B-factors for all atoms in the residue
            for atom in residue:
                residue_bfactors[residue_id].append(atom.get_bfactor())
        else:
            # If not in contact, add a zero for the residue
            residue_bfactors[residue_id].append(0)

    # Calculate average B-factor per residue (averaging over all atoms within a residue)
    avg_bfactors_per_residue = {res_id: sum(bfactors)/len(bfactors) 
                                for res_id, bfactors in residue_bfactors.items()}
    
    # Calculate the overall average (Direct average of all atoms in contact and non-contact residues)
    all_bfactors = [bfactor for bfactors in residue_bfactors.values() for bfactor in bfactors]
    overall_avg_bfactor = sum(all_bfactors) / len(all_bfactors) if all_bfactors else 0
    
    # Calculate the average of residue averages
    avg_bfactor_from_residues = sum(avg_bfactors_per_residue.values()) / len(avg_bfactors_per_residue) if avg_bfactors_per_residue else 0
    
    return overall_avg_bfactor, avg_bfactor_from_residues



def _load_model(model_name=''):
    if model_name.endswith(".pt"):  # local, treat as filepath
        model_path = Path(model_name)
        model_data = torch.load(str(model_path), map_location="cpu")
        
    else:  # load from hub
        url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
        model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")

    print('model load done')
    cfg = model_data["cfg"]["model"]
    model_state = model_data["model"]
    model = ESMFold(esmfold_config=cfg)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)

    return model




model = _load_model(model_name='data/esmfold_3B_v1.pt')

model = model.eval().cuda()
# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# 
# 
model.set_chunk_size(128)



gene_to_seq = dict()

gene_to_seq['tgfb1'] = 'MPPSGLRLLPLLLPLLWLLVLTPGRPAAGLSTCKTIDMELVKRKRIEAIRGQILSKLRLASPPSQGEVPPGPLPEAVLALYNSTRDRVAGESAEPEPEPEADYYAKEVTRVLMVETHNEIYDKFKQSTHSIYMFFNTSELREAVPEPVLLSRAELRLLRLKLKVEQHVELYQKYSNNSWRYLSNRLLAPSDSPEWLSFDVTGVVRQWLSRGGEIEGFRLSAHCSCDSRDNTLQVDINGFTTGRRGDLATIHGMNRPFLLLMATPLERAQHLQSSRHRRALDTNYCFSSTEKNCCVRQLYIDFRKDLGWKWIHEPKGYHANFCLGPCPYIWSLDTQYSKVLALYNQHNPGASAAPCCVPQALEPLPIVYYVGRKPKVEQLSNMIVRSCKCS'
gene_to_seq['tgfbr2'] = 'MGRGLLRGLWPLHIVLWTRIASTIPPHVQKSVNNDMIVTDNNGAVKFPQLCKFCDVRFSTCDNQKSCMSNCSITSICEKPQEVCVAVWRKNDENITLETVCHDPKLPYHDFILEDAASPKCIMKEKKKPGETFFMCSCSSDECNDNIIFSEEYNTSNPDLLLVIFQVTGISLLPPLGVAISVIIIFYCYRVNRQQKLSSTWETGKTRKLMEFSEHCAIILEDDRSDISSTCANNINHNTELLPIELDTLVGKGRFAEVYKAKLKQNTSEQFETVAVKIFPYEEYASWKTEKDIFSDINLKHENILQFLTAEERKTELGKQYWLITAFHAKGNLQEYLTRHVISWEDLRKLGSSLARGIAHLHSDHTPCGRPKMPIVHRDLKSSNILVKNDLTCCLCDFGLSLRLDPTLSVDDLANSGQVGTARYMAPEVLESRMNLENVESFKQTDVYSMALVLWEMTSRCNAVGEVKDYEPPFGSKVREHPCVESMKDNVLRDRGRPEIPSFWLNHQGIQMVCETLTECWDHDPEARLTAQCVAERFSELEHLDRLSGRSCSEEKIPEDGSLNTTK'

gene_to_seq['ccl19'] = 'MALLLALSLLVLWTSPAPTLSGTNDAEDCCLSVTQKPIPGYIVRNFHYLLIKDGCRVPAVVFTTLRGRQLCAPPDQPWVERIIQRLQRTSAKMKRRSS'
gene_to_seq['ccr7'] = 'MDLGKPMKSVLVVALLVIFQVCLCQDEVTDDYIGDNTTVDYTLFESLCSKKDVRNFKAWFLPIMYSIICFVGLLGNGLVVLTYIYFKRLKTMTDTYLLNLAVADILFLLTLPFWAYSAAKSWVFGVHFCKLIFAIYKMSFFSGMLLLLCISIDRYVAIVQAVSAHRHRARVLLISKLSCVGIWILATVLSIPELLYSDLQRSSSEQAMRCSLITEHVEAFITIQVAQMVIGFLVPLLAMSFCYLVIIRTLLQARNFERNKAIKVIIAVVVVFIVFQLPYNGVVLAQTVANFNITSSTCELSKQLNIAYDVTYSLACVRCCVNPFLYAFIGVKFRNDLFKLFKDLGCLSQEQLRQWSSCRHIRRSSMSVEAETTTTFSP'

gene_to_seq['cxcl12'] = 'MNAKVVVVLVLVLTALCLSDGKPVSLSYRCPCRFFESHVARANVKHLKILNTPNCALQIVARLKNNNRQVCIDPKLKWIQEYLEKALNKRFKM'
gene_to_seq['cxcr4'] = 'MEGISIYTSDNYTEEMGSGDYDSMKEPCFREENANFNKIFLPTIYSIIFLTGIVGNGLVILVMGYQKKLRSMTDKYRLHLSVADLLFVITLPFWAVDAVANWYFGNFLCKAVHVIYTVNLYSSVLILAFISLDRYLAIVHATNSQRPRKLLAEKVVYVGVWIPALLLTIPDFIFANVSEADDRYICDRFYPNDLWVVVFQFQHIMVGLILPGIVILSCYCIIISKLSHSKGHQKRKALKTTVILILAFFACWLPYYIGISIDSFILLEIIKQGCEFENTVHKWISITEALAFFHCCLNPILYAFLGAKFKTSAQHALTSVSRGSSLKILSKGKRGGHSSVSTESESSSFHSS'

gene_to_seq['mdk'] = 'MQHRGFLLLTLLALLALTSAVAKKKDKVKKGGPGSECAEWAWGPCTPSSKDCGVGFREGTCGAQTQRIRCRVPCNWKKEFGADCKYKFENWGACDGGTGTKVRQGTLKKARYNAQCQETIRVTKPCTPKTKAKAKAKKGKGKD'
gene_to_seq['ncl'] = 'MVKLAKAGKNQGDPKKMAPPPKEVEEDSEDEEMSEDEEDDSSGEEVVIPQKKGKKAAATSAKKVVVSPTKKVAVATPAKKAAVTPGKKAAATPAKKTVTPAKAVTTPGKKGATPGKALVATPGKKGAAIPAKGAKNGKNAKKEDSDEEEDDDSEEDEEDDEDEDEDEDEIEPAAMKAAAAAPASEDEDDEDDEDDEDDDDDEEDDSEEEAMETTPAKGKKAAKVVPVKAKNVAEDEDEEEDDEDEDDDDDEDDEDDDDEDDEEEEEEEEEEPVKEAPGKRKKEMAKQKAAPEAKKQKVEGTEPTTAFNLFVGNLNFNKSAPELKTGISDVFAKNDLAVVDVRIGMTRKFGYVDFESAEDLEKALELTGLKVFGNEIKLEKPKGKDSKKERDARTLLAKNLPYKVTQDELKEVFEDAAEIRLVSKDGKSKGIAYIEFKTEADAEKTFEEKQGTEIDGRSISLYYTGEKGQNQDYRGGKNSTWSGESKTLVLSNLSYSATEETLQEVFEKATFIKVPQNQNGKSKGYAFIEFASFEDAKEALNSCNKREIEGRAIRLELQGPRGSPNARSQPSKTLFVKGLSEDTTEETLKESFDGSVRARIVTDRETGSSKGFGFVDFNSEEDAKAAKEAMEDGEIDGNKVTLDWAKPKGEGGFGGRGGGRGGFGGRGGGRGGRGGFGGRGRGGFGGRGGFRGGRGGGGDHKPQGKKTKFE'

gene_to_seq['tifab'] = 'MEKPLTVLRVSLYHPTLGPSAFANVPPRLQHDTSPLLLGRGQDAHLQLQLPRLSRRHLSLEPYLEKGSALLAFCLKALSRKGCVWVNGLTLRYLEQVPLSTVNRVSFSGIQMLVRVEEGTSLEAFVCYFHVSPSPLIYRPEAEETDEWEGISQGQPPPGSG'
gene_to_seq['traf6'] = 'MSLLNCENSCGSSQSESDCCVAMASSCSAVTKDDSVGGTASTGNLSSSFMEEIQGYDVEFDPPLESKYECPICLMALREAVQTPCGHRFCKACIIKSIRDAGHKCPVDNEILLENQLFPDNFAKREILSLMVKCPNEGCLHKMELRHLEDHQAHCEFALMDCPQCQRPFQKFHINIHILKDCPRRQVSCDNCAASMAFEDKEIHDQNCPLANVICEYCNTILIREQMPNHYDLDCPTAPIPCTFSTFGCHEKMQRNHLARHLQENTQSHMRMLAQAVHSLSVIPDSGYISEVRNFQETIHQLEGRLVRQDHQIRELTAKMETQSMYVSELKRTIRTLEDKVAEIEAQQCNGIYIWKIGNFGMHLKCQEEEKPVVIHSPGFYTGKPGYKLCMRLHLQLPTAQRCANYISLFVHTMQGEYDSHLPWPFQGTIRLTILDQSEAPVRQNHEEIMDAKPELLAFQRPTIPRNPKGFGYVTFMHLEALRQRTFIKDDTLLVRCEVSTRFDMGSLRREGFQPRSTDAGV'

gene_to_seq['postn'] = 'MIPFLPMFSLLLLLIVNPINANNHYDKILAHSRIRGRDQGPNVCALQQILGTKKKYFSTCKNWYKKSICGQKTTVLYECCPGYMRMEGMKGCPAVLPIDHVYGTLGIVGATTTQRYSDASKLREEIEGKGSFTYFAPSNEAWDNLDSDIRRGLESNVNVELLNALHSHMINKRMLTKDLKNGMIIPSMYNNLGLFINHYPNGVVTVNCARIIHGNQIATNGVVHVIDRVLTQIGTSIQDFIEAEDDLSSFRAAAITSDILEALGRDGHFTLFAPTNEAFEKLPRGVLERIMGDKVASEALMKYHILNTLQCSESIMGGAVFETLEGNTIEIGCDGDSITVNGIKMVNKKDIVTNNGVIHLIDQVLIPDSAKQVIELAGKQQTTFTDLVAQLGLASALRPDGEYTLLAPVNNAFSDDTLSMDQRLLKLILQNHILKVKVGLNELYNGQILETIGGKQLRVFVYRTAVCIENSCMEKGSKQGRNGAIHIFREIIKPAEKSLHEKLKQDKRFSTFLSLLEAADLKELLTQPGDWTLFVPTNDAFKGMTSEEKEILIRDKNALQNIILYHLTPGVFIGKGFEPGVTNILKTTQGSKIFLKEVNDTLLVNELKSKESDIMTTNGVIHVVDKLLYPADTPVGNDQLLEILNKLIKYIQIKFVRGSTFKEIPVTVYTTKIITKVVEPKIKVIEGSLQPIIKTEGPTLTKVKIEGEPEFRLIKEGETITEVIHGEPIIKKYTKIIDGVPVEITEKETREERIITGPEIKYTRISTGGGETEETLKKLLQEEVTKVTKFIEGGDGHLFEDEEIKRLLQGDTPVRKLQANKKVQGSRRRLREGRSQ'
gene_to_seq['cd47'] = 'MWPLVAALLLGSACCGSAQLLFNKTKSVEFTFCNDTVVIPCFVTNMEAQNTTEVYVKWKFKGRDIYTFDGALNKSTVPTDFSSAKIEVSQLLKGDASLKMDKSDAVSHTGNYTCEVTELTREGETIIELKYRVVSWFSPNENILIVIFPIFAILLFWGQFGIKTLKYRSGGMDEKTIALLVAGLVITVIVIVGAILFVPGEYSLKNATGLGLIVTSTGILILLHYYVFSTAIGLTSFVIAILVIQVIAYILAVVGLSLCIAACIPMHGPLLISGLSILALAQLLGLVYMKFVASNQKTIQPPRKAVEEPLNAFKESKGMMNDE'

gene_to_seq['gdf15'] = 'MPGQELRTVNGSQMLLVLLVLSWLPHGGALSLAEASRASFPGPSELHSEDSRFRELRKRYEDLLTRLRANQSWEDSNTDLVPAPAVRILTPEVRLGSGGHLHLRISRAALPEGLPEASRLHRALFRLSPTASRSWDVTRPLRRQLSLARPQAPALHLRLSPPPSQSDQLLAESSSARPQLELHLRPQAARGRRRARARNGDHCPLGPGRCCRLHTVRASLEDLGWADWVLSPREVQVTMCIGACPSQFRAANMHAQIKTSLHRLKPDTVPAPCCVPASYNPMVLIQKTDTGVSLQTYDDLLAKDCHCI'
gene_to_seq['cd99'] = 'MARGAALALLLFGLLGVLVAAPDGGFDLSDALPDNENKKPTAIPKKPSAGDDFDLGDAVVDGENDDPRPPNPPKPMPNPNPNHPSSSGSFSDADLADGVSGGEGKGGSDGGGSHRKEGEEADAPGVIPGIVGAVVVAVAGAISSFIAYQKKKLCFKENAEQGEVDMESHRNANAEPAVQRTLLEK'

#sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
# Multimer prediction can be done with chains separated by ':'
#sequence = ccl19 + ':' + ccr7 # 76
#sequence = tgfb1 + ':' + tgfbr2 # 69
#sequence = cxcl12 + ':' + cxcr4 # 74
# sequence = mdk + ':' + ncl # 67
#sequence = tifab + ':' + traf6 # 72 and 72
#sequence = gdf15 + ':' + cd99 # 66
#sequence = cd47 + ':' + postn # 74
ligand = ['ccl19', 'tgfb1', 'cxcl12', 'mdk', 'tifab', 'gdf15', 'postn']
receptor = ['ccr7', 'tgfbr2', 'cxcr4', 'ncl', 'traf6', 'cd99', 'cd47']

sequence_list = []
for i in range (0, len(ligand)):
    sequence_list.append(gene_to_seq[receptor[i]] + ':' + gene_to_seq[ligand[i]])

for i in range(6, len(sequence_list)):
    sequence = sequence_list[i]
    print('\n%s - %s'%(ligand[i], receptor[i]))
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open("result.pdb", "w") as f:
        f.write(output)

    #import biotite.structure.io as bsio
    #struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
    #print(struct.b_factor.mean())  # this will be the pLDDT
    # 88.3

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", "result.pdb")

    bfactors = []
    for atom in structure.get_atoms():
        bfactors.append(atom.bfactor)

    print("Mean pLDDT:", sum(bfactors)/len(bfactors))


    overall_avg, avg_residue = calculate_bfactor_for_contact_residues_with_zeros("result.pdb", "A", "B")
    print(f"Overall weighted average B-factor for chain B: {overall_avg}")
    print(f"Weighted average of residue average B-factors for chain B: {avg_residue}")


####################################################################################
def min_chain_distance(pdb_file, chain1="A", chain2="B"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_file)

    atoms1 = [atom.get_coord() for atom in structure[0][chain1].get_atoms() if atom.element != "H"]
    atoms2 = [atom.get_coord() for atom in structure[0][chain2].get_atoms() if atom.element != "H"]

    atoms1 = np.array(atoms1)
    atoms2 = np.array(atoms2)

    # pairwise distances
    dists = np.sqrt(((atoms1[:, None, :] - atoms2[None, :, :]) ** 2).sum(axis=-1))
    return dists.min()

d = min_chain_distance("result.pdb", "A", "B")
print("Min distance (Å):", d)


#######################################################




#202511264030