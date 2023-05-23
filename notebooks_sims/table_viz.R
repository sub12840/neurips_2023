# R-script to get Latex tables, does not have any other purpose
library(tidyverse)
library(stargazer)
home_string = ''
setwd(home_string)

data_metrics <- read_csv('data/export_all_metrics.csv', col_names = F, skip=2)
names(data_metrics) <- c('metric', sort(c(paste(c('Boosting', 'Lasso', 'Oneto'),'mean'),
                                          paste(c('Boosting', 'Lasso', 'Oneto'),'std') )))

data_metrics %>% 
  filter(str_detect(metric, 'param')) %>% 
  select(-c(contains('Oneto'))) %>% 
  {.->> check_1 } %>% 
  mutate(across(where(is.numeric), round, 2)) %>% 
  mutate(is_semi = if_else(str_detect(metric, 'semi'), 1, 0)) %>% 
  gather(key = var_name, value = value, 2:ncol(check_1)) %>% 
  spread(key = names(check_1)[1], value = 'value') %>% 
  filter(var_name != 'is_semi') %>% 
  select(var_name, mse_1_param_semifair, mse_2_param_semifair,
         dist_param_semifair, budget_param_semifair, 
         mse_1_param_fair,mse_2_param_fair,
         dist_param_fair, budget_param_fair) %>% 
  stargazer(summary=FALSE, rownames = F)

data_metrics %>% 
  filter(!str_detect(metric, 'param')) %>% 
  {.->> check_2} %>% 
  mutate(across(where(is.numeric), round, 2)) %>% 
  gather(key = var_name, value = value, 2:ncol(check_2)) %>% 
  spread(key = names(check_2)[1], value = 'value') %>% 
  select(var_name,
         mse_1_fair, mse_2_fair, dist_fair, budget_fair) %>% 
  stargazer(summary=FALSE, rownames = F)

