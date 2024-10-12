
import modulus.sym
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.key import Key

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator
from modulus.sym.dataset import HDF5GridDataset

from modulus.sym.utils.io.plotter import GridValidatorPlotter

#from utilities import download_FNO_dataset
from utilities import download_FNO_dataset

@modulus.sym.main(config_path= "./", config_name="config_FNO.yaml")

def run(cfg: ModulusConfig) -> None:
    input_keys = [Key(("coeff"), scale=(7.48360e00, 4.49996e00))]
    output_keys = [Key(("sol"), scale=(5.74634e-03, 3.88433e-03))]
    
    download_FNO_dataset("Darcy_241", outdir="datasets/")
    
    train_path = to_absolute_path("datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5")
    test_path = to_absolute_path("datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5")
    
    train_dataset = HDF5GridDataset(
        train_path, invar_keys = ["coeff"], outvar_keys=["sol"], n_examples= 1000
    )
    
    test_dataset = HDF5GridDataset(
        test_path, invar_keys = ["coeff"], outvar_keys=["sol"], n_examples= 100
    )
    
    decoder_net = instantiate_arch(
        cfg = cfg.arch.decoder,
        output_keys = output_keys,
    )
    
    fno = instantiate_arch(
        cfg = cfg.arch.fno,
        input_keys = input_keys,
        decoder_net = decoder_net,
    )
    
    nodes = [fno.make_node("fno")]
    
    domain = Domain()
    
    supervised = SupervisedGridConstraint(
        nodes = nodes,
        dataset = train_dataset,
        batch_size = cfg.batch_size.grid,
        num_workers = 4,
    )
    domain.add_constraint(supervised, "supervised")
    
    val = GridValidator(
        nodes = nodes,
        dataset = test_dataset,
        batch_size = cfg.batch_size.validation,
        plotter = GridValidatorPlotter(n_examples = 10)
    )
    domain.add_validator(val, "val")
    
    slv = Solver(cfg, domain)
    
    slv.solve()
    
    
    
if __name__ == "__main__":
    run()