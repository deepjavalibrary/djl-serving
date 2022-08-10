import Vue from 'vue'
import Router from 'vue-router'


const Home = () => import('@/view/Home.vue')
const Main = () => import('@/view/Main.vue')
const ModelList = () => import('@/view/ModelList.vue')
const AddModel = () => import('@/view/AddModel.vue')
const UpdateModel = () => import('@/view/UpdateModel.vue')
const Inference = () => import('@/view/Inference.vue')
const Log = () => import('@/view/Log.vue')


Vue.use(Router)

const router = new Router({
  routes: [
    {
      path: '/',
      redirect:"home",
      component: Main,
      children: [
        { path: '/home', component: ModelList, },
        { path: '/model-list', component: ModelList, },
        { path: '/add-model', component: AddModel, },
        { path: '/update-model/:name', component: UpdateModel, },
        { path: '/inference/:name', component: Inference, },
        { path: '/log', component: Log, },
      ]
    }
  ]
})


export default router
