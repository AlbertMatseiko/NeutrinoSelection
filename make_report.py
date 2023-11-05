import tensorflow as tf
import numpy as np
from IPython.display import clear_output
import os
from analysis import Analysis as A

# data and model's names
data_names = [n for n in os.listdir('data/') if n.endswith('.h5')]
for i, h5n in enumerate(data_names):
    print(str(i + 1), ". " + h5n)
i = int(input("Which dataset do you want to use? Print it's number! \n"))
name = data_names[i - 1]
path_to_h5 = './data/' + name

model_names = [n for n in os.listdir('trained_models/') if not n.startswith('logs') and not n.startswith('.')]
for i, mn in enumerate(model_names):
    print(str(i + 1), ". " + mn)
i = int(input("Which model do you want to choose? Print it's number! \n"))
model_name = model_names[i - 1]

# Initializing analysis class and predicions
trigger = input("Do you want to create new predictions for 'val' regime? Type only 'y' or 'n': \n")
if trigger == 'y':
    path_to_model = './trained_models/' + model_name + '/' + 'best'  # Change to best later!
    model = tf.keras.models.load_model(path_to_model, compile=False)
    model._name = model_name
    an = A(model=model,
           model_name=model_name,
           path_to_h5=path_to_h5,
           regime='val')
    clear_output(wait=False)
    _ = an.make_preds()
elif trigger == 'n':
    an = A(model=None,
           model_name=model_name,
           path_to_h5=path_to_h5,
           regime='val')
    pass
else:
    an = A(model=None,
           model_name=model_name,
           path_to_h5=path_to_h5,
           regime='val')
    print("Your input is incorrect. Preds will not be recreated.")

# create report dir
path_to_report = './preds_analysis/report_' + model_name + '_' + name[:-3]
try:
    os.makedirs(path_to_report)
    print('directory for report is created')
except:
    print('directory for report already exists')

an.load_preds_and_labels()
an.separate_preds()
print("All predictions are loaded!")

### SOMETHING GLOBAL ###
an.N_points = 10000
an.tr_start, an.tr_end = 0., 1.
update = True #False

# Get S and E
an.get_pos_rates(batch_size=2048 * 64, update_positives=update)
fig = an.plot_SE()
postfix = f"_{an.N_points}_{an.tr_start}_{an.tr_end}"
fig.write_html(path_to_report + "/fig_SE" + postfix + ".html")
print("Picture of S and E is created!")
i_crit = np.argwhere(an.S <= 1e-6)[0]
report_file = open(path_to_report + "/info_ES.txt", "w")
tr_crit = i_crit / an.S.shape[0] * (an.tr_end - an.tr_start)
report_file.write("Critical treshold = " + str(tr_crit) + '\n')
report_file.write("Level of suppression = " + str(an.S[i_crit]) + '\n')
report_file.write("Level of exposition = " + str(an.E[i_crit]) + '\n')
report_file.close()
print("Critical treshold =", tr_crit)
print("Level of suppression =", an.S[i_crit])
print("Level of exposition =", an.E[i_crit])

# Flux algorithm
nu_in_flux = 50
mu_nu_ratio = 1e5
start_mu = int(1e6)
start_nu = int(1e6)
an.get_NuFromNN(nu_in_flux=nu_in_flux,
                mu_nu_ratio=mu_nu_ratio,
                start_mu=start_mu, start_nu=start_nu,
                update_positives=update)
print("Positives for flux are extracted!")
print('Mu number =', mu_nu_ratio * nu_in_flux)
print('True Nu number =', nu_in_flux)
postfix = f"_{nu_in_flux}_{mu_nu_ratio}_{start_mu}_{start_nu}"
fig_flux = an.plot_error_and_flux()
fig_flux.write_html(path_to_report + "/fig_flux" + postfix + ".html")
report_file = open(path_to_report + "/last_info_flux" + postfix + ".txt", "w")
report_file.write(
    f"Mu number = {an.nu_in_flux * an.mu_nu_ratio}\nNu number = {an.nu_in_flux}\nStarts of slices = {start_mu}, {start_nu}.")
report_file.close()
