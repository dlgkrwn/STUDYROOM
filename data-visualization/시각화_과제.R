library(ggplot2)
library(gapminder)
library(socviz)
mtcars
#1번
mtcars
mtcars$car_name = rownames(mtcars)
mtcars
p50 <- ggplot(data = mtcars, mapping = aes(x = car_name, y = mpg))
p50 + geom_col(fill = "blue") +
  xlab("car_names") +
  theme(axis.text.x=element_text(angle=45))

#2번
mtcars
p51 <- ggplot(data = mtcars, mapping = aes(x = reorder(car_name, mpg), y = mpg))
p51 + geom_col(width=0.1,color="blue") +
  geom_point(color = "green", size = 5) +
  coord_flip() + xlab("car_names")

#3번
p52 <- ggplot(data = mtcars, mapping = aes(x = wt,
                                           y = mpg))
p52 + geom_point() + geom_smooth(method = "lm")

#4번
p53 <- ggplot(data=mtcars, aes(x=wt,y=mpg))
p53 + geom_point(aes(color=as.factor(am))) + 
  labs(title="차량 수동/자동 비교")

#5번
diamonds
p54 <- ggplot(data = diamonds, mapping = aes(x = carat, y = price, group = color))
p54 + geom_point(aes(color = color)) + geom_smooth(method = "gam")

#6번
p55 <- ggplot(data = gapminder, mapping = aes(x = gdpPercap, y = lifeExp))
p55 + geom_point() + geom_smooth(method = 'lm') +
      scale_x_log10()


#7번
p56 <- ggplot(data = gapminder,
              mapping = aes(x = gdpPercap,
                            y = lifeExp,
                            color = continent,
                            fill = continent))
p56 + geom_point() + geom_smooth(method = "loess") +
  scale_x_log10()

#8번
diamonds
p57 <- ggplot(data = diamonds, mapping = aes(x = cut,
                                           fill = color))
p57 + geom_bar(position = "dodge", mapping = aes(y = ..prop.., group = color))

#9번
data(gss_sm)
temp <-gss_sm %>%
  group_by(bigregion, religion) %>%
  summarize(N=n()) %>%
  mutate(freq=N/sum(N), pct=round(N/sum(N)*100,1))
temp

#10번
p59 <- ggplot(data = gss_sm, mapping = aes(y = religion, fill = religion))
p59 + geom_bar(position = "dodge", mapping = aes(group = religion)) +
       facet_wrap(~bigregion, ncol = 4) + guides(fill = FALSE)








